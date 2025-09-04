---
title: bitnet中int2和int8的使用
date: 2025-08-24
authors: [KenForever1]
categories: 
  - cpp
labels: [cpp]
pin: true
comments: true
---

<!-- more -->

## bitnet中int2和int8的使用

W2A8‌（2-bit权重 + 8-bit激活）是一种极低精度量化方案，旨在通过牺牲部分模型精度换取显著的存储和计算效率提升。适用于计算资源受限的场景边缘设备部署‌，如移动端或嵌入式设备‌。

### gpu版本推理

[参考readme中](https://github.com/microsoft/BitNet/tree/main/gpu)使用的 BitNet-b1.58-2B-4T模型。

权重weight采用int2存储，激活activation采用int8存储。自定义了个cuda kernel函数，用于计算int2和int8的乘法。

+ 在转换checkpoint时，将int8权重转为int2（convert_weight_int8_to_int2），并且对权重进行了划分为16×32 blocks和重排，便于gpu的高效运算。

+ 自定义kernel函数ladder_int8xint2_kernel，采用gpu dp4a指令(__dp4a)进行低精度的点积操作。

![](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/10/DP4A_DP2A-624x223.png)


在prefill时，使用fp16；在decode时，使用int2，采用自定义cuda kernel函数进行int2和int8的乘法。
```python
# \gpu\generate.py
model_args_prefill = fast.ModelArgs(use_kernel=False)
model_args_decode = fast.ModelArgs(use_kernel=True)
tokenizer = Tokenizer("./tokenizer.model")

prefill_model = fast.Transformer(model_args_prefill)
decode_model = fast.Transformer(model_args_decode)

fp16_ckpt_path = str(Path(ckpt_dir) / "model_state_fp16.pt")
fp16_checkpoint = torch.load(fp16_ckpt_path, map_location="cpu")
int2_ckpt_path = str(Path(ckpt_dir) / "model_state_int2.pt")
int2_checkpoint = torch.load(int2_ckpt_path, map_location="cpu")
prefill_model.load_state_dict(fp16_checkpoint, strict=True)
decode_model.load_state_dict(int2_checkpoint, strict=True)
```
ladder_int8xint2_kernel核函数用于计算int2乘int8的向量乘法。
```c++
template <int M, int N, int K, int ws_num, int K_block_size, int N_block_size>
__global__ void __launch_bounds__(128) ladder_int8xint2_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ dtype_transform, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {
  constexpr int K_per_loop = 16;
  constexpr int wmma_K = 32;
  constexpr int wmma_N = 16;
  int in_thread_C_local[1];
  signed char A_local[K_per_loop];
  int B_reshape_local[1];
  signed char B_decode_local[K_per_loop];
  int red_buf0[1];
  in_thread_C_local[0] = 0;
  #pragma unroll
  for (int k_0 = 0; k_0 < K/(K_per_loop * K_block_size); ++k_0) {
    *(int4*)(A_local + 0) = *(int4*)(A + ((k_0 * K_per_loop * K_block_size) + (((int)threadIdx.x) * K_per_loop)));
    B_reshape_local[0] = *(int*)(B + 
      (((int)blockIdx.x) * N_block_size * K / 4) + 
      (k_0 * K_block_size * K_per_loop * wmma_N / 4) +
      ((((int)threadIdx.x) >> 1) * wmma_K * wmma_N / 4) +
      ((((int)threadIdx.y) >> 3) * (wmma_K * wmma_N / 2) / 4) + 
      ((((int)threadIdx.x) & 1) * (wmma_K * wmma_N / 4) / 4) + 
      ((((int)threadIdx.y) & 7) * (wmma_K / 2) / 4)
      );
    decode_i2s_to_i8s(B_reshape_local, B_decode_local, 16);
    #pragma unroll
    for (int k_2_0 = 0; k_2_0 < 4; ++k_2_0) {
      in_thread_C_local[0] = __dp4a(*(int *)&A_local[((k_2_0 * 4))],*(int *)&B_decode_local[((k_2_0 * 4))], in_thread_C_local[0]);
    }
  }
  red_buf0[0] = in_thread_C_local[0];
  #pragma unroll
  for (int offset = K_block_size/2; offset > 0; offset /= 2) {
    red_buf0[0] += __shfl_down_sync(__activemask(), red_buf0[0], offset, K_block_size);
  }
  int out_idx = ((((int)blockIdx.x) * N_block_size) + ((int)threadIdx.y));
  int ws_idx = out_idx / (N / ws_num);
  if (threadIdx.x == 0)
    dtype_transform[out_idx] = (__nv_bfloat16)(((float)red_buf0[0])/(float)s[0]*(float)ws[ws_idx]);
}
```

自定义kernel NVIDIA GeForce RTX 4050性能测试运行结果：
```
custom == np True
Shape(2560, 2560), W2A8: 54.40us, torch BF16: 73.39us
custom == np True
Shape(3840, 2560), W2A8: 22.66us, torch BF16: 75.50us
custom == np True
Shape(13824, 2560), W2A8: 39.99us, torch BF16: 364.94us
custom == np True
Shape(2560, 6912), W2A8: 48.95us, torch BF16: 221.11us
custom == np True
Shape(3200, 3200), W2A8: 45.98us, torch BF16: 128.34us
custom == np True
Shape(4800, 3200), W2A8: 25.54us, torch BF16: 171.50us
custom == np True
Shape(3200, 10240), W2A8: 37.18us, torch BF16: 342.03us
custom == np True
Shape(20480, 3200), W2A8: 74.29us, torch BF16: 805.49us
```

### cpu版本推理

cpu推理基于llama.cpp项目，采用T-MAC(查找表)算法提高了推理性能。

传统方式，需要把权重反量化为浮点数，然后和激活进行计算。采用查找表就是，把1bit的各种计算可能提前计算好存储到一个查找表中，然后通过索引进行查找完成计算。

#### SIMD计算int2和int8的向量点积

bitnet的实现分为intel的AVX 和 ARM 的 neon 指令， 这里对[AVX指令版本](https://github.com/microsoft/BitNet/blob/main/src/ggml-bitnet-mad.cpp)进行介绍。

将量化后的 i2 数据与 i8 数据做点积，利用平台 SIMD 指令做并行加速。
为了高效处理，数据按组分块，每组做并行处理，最后合并结果。
针对不同平台分别优化（AVX2/NEON）。
```c++
for each block of 32 groups:
    for each group in block:
        1. 提取2-bit数据
        2. 载入8-bit数据
        3. 做乘法累加（SIMD并行）
    累加所有组的结果
最后合并所有块的结果，输出点积
```

```c++
// 1. 初始化掩码和累加器
// mask = _mm256_set1_epi8(0x03); 用于提取2-bit数据。
// accu = _mm256_setzero_si256(); 用于累加结果。
// 2. 分组处理
// 外层循环：每次处理32组。
// 内层循环：遍历每组中的32个元素。
// 对每一组数据，分别取出2-bit并与8-bit数据做乘法累加（点积）。
// 3. 数据解码
// 2-bit数据拆成4个bit组（shift和mask操作）。
// 8-bit直接载入。
// 使用 _mm256_maddubs_epi16 指令做乘法累加。
// 4. 累加所有结果
// 使用 _mm256_add_epi16 和 _mm256_add_epi32 指令合并结果。
// 最后用 hsum_i32_8 对 SIMD 累加结果求和，得到最终点积。

*s = (float)sumi;，将结果赋值给输出指针。

#include <vector>
#include <type_traits>

// #include "ggml-bitnet.h"
// #include "ggml-quants.h"
#include <cmath>
#include <cstring>

#define QK_I2_S 128
#define QK_I2 128

#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}


void print_m256i(__m256i vec) {
    uint8_t bytes[32];
    _mm256_storeu_si256((__m256i*)bytes, vec);
    for (int i = 0; i < 32; ++i) {
        printf("%02x ", bytes[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
}

// x 指向 2-bit 量化数据，y 指向 8-bit 数据。
// 数据被分成多个组，便于后面的 SIMD 并行处理。
// 主要变量：
// QK_I2_S 通常为 128，表示每组的元素数。
// nb = n / QK_I2_S，组数。
// group32_num = nb / 32，每 32 组为一大组。
// la_num = nb % 32，剩余组数。
void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    // n : 128 nb : 1 group32_num : 0 la_num : 1 groupla_num : 1
    std::cout << "n : " << n
              << " nb : " << nb
              << " group32_num : " << group32_num
              << " la_num : " << la_num
              << " groupla_num : " << groupla_num
              << std::endl;

    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i accu = _mm256_setzero_si256();

    for (int i=0; i < group32_num; i++){
        std::cout << "i : " << i << std::endl;
        __m256i accu32 = _mm256_setzero_si256();
        for (int j=0; j < 32; j++) {
        // 128 index
        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + i * 32 * 32 + j * 32));
        __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
        __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
        __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

        // each 32 index
        xq8_3 = _mm256_and_si256(xq8_3, mask);
        xq8_2 = _mm256_and_si256(xq8_2, mask);
        xq8_1 = _mm256_and_si256(xq8_1, mask);
        xq8_0 = _mm256_and_si256(xq8_0, mask);

        // each 32 index
        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 96));

        // 128 index accumulation add
        // split into 32 accumulation block
        // each block each 128 index accumulated 4index
        // each index maximum 256
        // each block maximum 4 * 256
        // each block accumulation maximum 127 * 256
        // each 32 group index (128 index in one group) needs cast to int32
        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_0, xq8_1));
        accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, _mm256_set1_epi16(1)), accu);
    }

    for (int i = 0; i < groupla_num; i++){
        __m256i accula = _mm256_setzero_si256();
        for (int j = 0; j < la_num; j++) {
        // 128 index
        // 例子中的128个int2，分成4组，每组32个int2
        // 128个int2
        // xx1 xx2 xx3 xx4 xx5 xx6 xx7 xx8 (int 16)
        // 00  xx1 xx2 xx3 xx4 xx5 xx6 xx7 (shift right 2bit)
        // 00  00  xx1 xx2 xx3 xx4 xx5 xx6 (shift right 4bit)
        // 00  00  00  xx1 xx2 xx3 xx4 xx5 (shift right 6bit)
        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + group32_num * 32 * 32 + j * 32));
        __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
        __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
        __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

        // each 32 index
        // mask: 00000011 00000011
        // --  --  --  xx4 -- -- -- xx8 (int 16)
        // 00  --  --  xx3 -- -- -- xx7 (shift right 2bit)
        // 00  00  --  xx2 -- -- -- xx6 (shift right 4bit)
        // 00  00  00  xx1 -- -- -- xx5 (shift right 6bit)
        // int2通过移动bit位置和mask，拆分成了int8表示，就可以直接和int8进行运算了
        xq8_3 = _mm256_and_si256(xq8_3, mask);
        xq8_2 = _mm256_and_si256(xq8_2, mask);
        xq8_1 = _mm256_and_si256(xq8_1, mask);
        xq8_0 = _mm256_and_si256(xq8_0, mask);


        print_m256i(xq8_3);
        printf("-----\n");
        print_m256i(xq8_2);
        printf("-----\n");
        print_m256i(xq8_1);
        printf("-----\n");
        print_m256i(xq8_0);
        printf("-----\n");
        
        // x中的int2每间隔4个数字为一组
        // 03 03 03 03 03 03 03 03 03 03 03 03 03 03 03 03 
        // 03 03 03 03 03 03 03 03 03 03 03 03 03 03 03 03 
        // -----
        // 02 02 02 02 02 02 02 02 02 02 02 02 02 02 02 02 
        // 02 02 02 02 02 02 02 02 02 02 02 02 02 02 02 02 
        // -----
        // 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 
        // 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 
        // -----
        // 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
        // 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00


        // each 32 index
        // 例子中的128个int8数据，分成4组，每组32个int8数据
        // xx1 xx2
        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 96));

        print_m256i(yq8_0);
        printf("-----\n");
        print_m256i(yq8_1);
        printf("-----\n");
        print_m256i(yq8_2);
        printf("-----\n");
        print_m256i(yq8_3);
        
        // 按顺序存储，相邻的32个int8为一组
        // -----
        // 00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 
        // 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f 
        // -----
        // 20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f 
        // 30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f 
        // -----
        // 40 41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f 
        // 50 51 52 53 54 55 56 57 58 59 5a 5b 5c 5d 5e 5f 
        // -----
        // 60 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f 
        // 70 71 72 73 74 75 76 77 78 79 7a 7b 7c 7d 7e 7f


        // 128 index accumulation add
        // split into 32 accumulation block
        // each block each 128 index accumulated 4index
        // each index maximum 256
        // each block maximum 4 * 256
        // each block accumulation maximum 127 * 256
        // each 32 group index (128 index in one group) needs cast to int32
        // 乘加运算
        // xx1, xx5 (16bit)
        // yy1, yy2 (16bit)
        // res = xx1 * yy1  + xx5 * yy2
        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        // printf("-----\n");
        // print_m256i(xq8_3);
        // printf("-----\n");
        // print_m256i(xq8_2);
        // printf("-----\n");、
        // 打印第1组乘加计算结果
        print_m256i(xq8_1);
        printf("-----\n");
        // print_m256i(xq8_0);
        // printf("-----\n");
        

        // 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 
        // 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01 01
        // 乘加计算
        // 20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f 
        // 30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
        // 结果
        // 41 00 45 00 49 00 4d 00 51 00 55 00 59 00 5d 00 
        // 61 00 65 00 69 00 6d 00 71 00 75 00 79 00 7d 00


        accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
        accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, _mm256_set1_epi16(1)));
    }
    int sumi = hsum_i32_8(accu);
    *s = (float)sumi;
}
```
测试代码：
```c++
// g++ int2_int8.cpp  -mavx2
// cat /proc/cpuinfo | grep avx
int main()
{
    bool is_debug = false;

    // 设定向量长度为128（一个block）
    int n = QK_I2_S;
    uint8_t i2_data[QK_I2_S / 4]; // 每4个2bit压缩成1字节，128个int2, 也就是_mm256_loadu_si256一次load256bit，要load 1次，一次load 128个int2数字，再通过偏移量分成32个int2一组，分成4组
    int8_t i8_data[QK_I2_S]; // 128个int8，也就是_mm256_loadu_si256一次load256bit,要load 4次，每次load32个int8数字

    // _mm256_maddubs_epi16就可以一次计算32个int8数字和int2数字

    // 构造简单数据：i2全部为1，i8全部为1
    // 2bit编码：2bit可表示4种状态（00, 01, 10, 11），若采用带符号整数表示，范围为-2到1；无符号整数范围为0到3
    for (int i = 0; i < QK_I2_S / 4; ++i) {
        if (!is_debug){
            // 每字节填充为 0x55 (01010101) 即每个2bit为"1"
            i2_data[i] = 0x55;
        }else{
            // 00 01 10 11
            i2_data[i] = 0x1B;
        }
    }
    for (int i = 0; i < QK_I2_S; ++i) {
        if (!is_debug){
            i8_data[i] = 1;
        }else{
            i8_data[i] = i;
        }
    }

    float result = 0.0f;
    ggml_vec_dot_i2_i8_s(n, &result, 0, i2_data, 0, i8_data, 0, 0);

    std::cout << "测试结果: " << result << std::endl;

    // 理论上，如果i2都解码为1，i8都为1，点积应为128
    if (result == 128.0f) {
        std::cout << "通过" << std::endl;
    } else {
        std::cout << "未通过" << std::endl;
    }
    return 0;
}
```

代码似乎没有完全开源，比如没有在代码中看到如下内容：
+ T-MAC的实现和应用？
+ ggml_vec_dot_i2_i8_s如何被调用上的？