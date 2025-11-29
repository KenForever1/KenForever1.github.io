---
title: 开发技法之pow计算simd优化
date: 2025-11-29
authors: [KenForever1]
categories: 
  - 技法
labels: []
comments: true
---

在阅读[vv](https://github.com/wolfpld/vv)终端图片查看器代码时，看到了作者加速的一个trick，效果提升明显，分享记录一下。

<!-- more -->

## 背景

问题：一张测试图像分辨率为9504×6336，加载时间超过了8秒。这是无法接受的。
优化方法：多线程、simd加速计算。

### 多线程

首先采用了多线程的方式，
所有计算都是针对单个像素的局部计算，不依赖于相邻像素，应该是极易并行化的。
分别对每个处理步骤进行并行化处理，以遵循串行的操作方式。然后，经过一番分析，我意识到这种做法效率很低。我使用的测试图像大小为9504×6336×4通道×4字节，即918MB。因此，并行化的色彩管理函数需要加载这918MB的数据，进行处理，再将918MB的数据存储回内存。接着，并行化的转换函数又需要再次加载这918MB的数据，进行必要的计算，然后再次存储918MB的数据。后续的步骤也是如此。难怪分析结果显示，一半的执行时间都花在了内存指令上，因为所有数据都必须不断地通过内存进行传输。

改进方法：
彻底改变加载器的工作方式。现在，并行处理在顶层进行，每个任务首先加载一部分YCbCr数据，然后在一个小的临时缓冲区中逐步完成所有必要的处理，最后将处理好的图像部分写入输出位图。这些数据块足够小，可以放入缓存中，而且在性能分析时，内存指令的问题也不再出现了。


### simd加速

在处理HDR图像时，需要使用PQ函数，需要有std::pow()函数的实现才能使用。

> Pq 实现的是 PQ (Perceptual Quantizer) 量化，这是 HDR (高动态范围) 图像处理中常用的非线性编码函数，用于将线性光信号转换为感知量化的非线性值。

作者做了一些尝试：
+ [已有的SSE2实现](https://github.com/JishinMaster/simd_utils/blob/master/sse_mathfun.h)（能同时进行3次pow()计算）实际上比串行实现还要慢。
+ 用fma（融合乘加）指令替代一系列的mul + add指令，略微让SIMD占据了优势。将实现扩展到AVX2带来了预期的2倍速度提升，但AVX512版本却产生了失真。后来我发现，在阅读SIMD文档时，我不知怎么漏掉了正确的舍入选项，用了错误的那个，导致输出出现了NaN。但这也无关紧要了，我对那个版本的性能并不满意。

然后，作者决定不再网上找代码了，“或许是时候停止尝试从网上随便找代码了，或许我真的应该弄明白这东西到底该怎么运行。”

先透露哈，实际上作者采用了数学优化方法去计算pow。将pow转换成了exp和log的组合。然后exp和log的计算采用了多项式逼近的方法，来减少乘法的次数，分别实现了SSE2、AVX、AVX512的版本优化。取得了巨大的性能提升。

#### 背景数学知识

1. 幂函数可以表示为exp()和log()函数的组合。

$ x^y = e^{y \cdot \ln x} $ 或者 $ x^y = 2^{y \cdot \log_2 x} $

下面对图像处理中浮点数计算的优化，采用$ x^y = 2^{y \cdot \log_2 x} $。不使用e的原因是减少不必要的数字来回转换。

2. IEEE 754浮点数的编码方式的构成：

+ 1 bit sign, S, 1位符号位S，

+ 8 bits exponent, E, 8位指数E，

+ 23 bits mantissa, M. 23位尾数M。

$ -1^S \cdot 1.M \cdot 2^{E - 127} $

由于是图片处理，符号位与我们无关，颜色值永远不会是负数。

E−127只是为了解码该数字的二进制编码，我们会将其简化为E，假设它有适当的偏移量。

尾数的值写作$ 1.M $， 总是在1到2的范围内。当1到2的范围值，乘以 $ 2^{E} $，范围就变成了0.5-1,2-4，等等。


然后根据另一个数学知识：

$ \log(a \cdot b) = \log a + \log b $

在图片处理的浮点数处理中，就可以做下面的转换：

$$ \log_2(1.M \cdot 2^E) = \log_2 1.M + \log_2 2^E $$

$$ \log_2(1.M \cdot 2^E) = \log_2 1.M + E $$

对于基数2，我们所要做的就是计算尾数的对数，再加上指数的值，就能得到整个浮点数的对数。而且，由于尾数的范围总是在1到2之间，我们可以用多项式函数相当精确地近似其对数。指数函数也可以通过多项式和一些类似的技巧来近似。


> 处理完幂函数后，将PQ变换实现为SIMD函数就相当简单了。该代码的标量版本运行时间为1.56秒。SSE 4.1 + FMA版本仅需528毫秒，速度提升约3倍。太棒了！扩展SIMD代码很简单，因为通道之间没有串扰，否则就需要代价高昂的混洗和置换操作。AVX2版本的运行时间为276毫秒，而AVX512版本仅需145毫秒。

> 通过我们已经讨论过的并行化处理，运行时间进一步缩短到仅31毫秒。这仅为原始1.56秒的2%！
>
> 优化前，测试图像加载需要8秒多。有了SIMD代码路径和多线程后，同一图像的加载时间仅为0.8秒。

## simd实现

### 原始PQ实现

PQ 量化基于人类视觉系统的感知特性，通过非线性映射将高动态范围的光线值压缩到有限的编码空间中，同时保持感知上的均匀性。可以用于HDR 图像显示，这个函数是 ST.2084 标准中定义的 PQ 曲线的实现，用于将线性光值转换为感知量化的电平值。
原始的PQ实现：
```c++
float Pq( float N )
{
    constexpr float m1 = 0.1593017578125f;
    constexpr float m1inv = 1.f / m1;
    constexpr float m2 = 78.84375f;
    constexpr float m2inv = 1.f / m2;
    constexpr float c1 = 0.8359375f;
    constexpr float c2 = 18.8515625f;
    constexpr float c3 = 18.6875f;

    const auto Nm2 = std::pow( std::max( N, 0.f ), m2inv );
    return 10000.f * std::pow( std::max( 0.f, Nm2 - c1 ) / ( c2 - c3 * Nm2 ), m1inv ) / 255.f;
}

void LinearizePq( float* ptr, int sz )
{
    for( int i=0; i<sz; i++ )
    {
        ptr[0] = Pq( ptr[0] );
        ptr[1] = Pq( ptr[1] );
        ptr[2] = Pq( ptr[2] );

        ptr += 4;
    }
}
```

### simd实现
这里只展示了__SSE4_1__的实现，256位和512位的操作可以查看源码。重点优化点是自定义实现了_mm_pow_ps函数。
```c++
#if defined __SSE4_1__ && defined __FMA__
void LinearizePq128( float* ptr, int sz )
{
    while( sz > 0 )
    {
        __m128 px0 = _mm_loadu_ps( ptr );
        __m128 px1 = _mm_max_ps( px0, _mm_setzero_ps() );
        __m128 Nm2 = _mm_pow_ps( px1, _mm_set1_ps( 1.f / 78.84375f ) );

        __m128 px2 = _mm_sub_ps( Nm2, _mm_set1_ps( 0.8359375f ) );
        __m128 px3 = _mm_max_ps( px2, _mm_setzero_ps() );

        __m128 px4 = _mm_fnmadd_ps( _mm_set1_ps( 18.6875f ), Nm2, _mm_set1_ps( 18.8515625f ) );
        __m128 px5 = _mm_div_ps( px3, px4 );

        __m128 px6 = _mm_pow_ps( px5, _mm_set1_ps( 1.f / 0.1593017578125f ) );
        __m128 ret = _mm_mul_ps( px6, _mm_set1_ps( 10000.f / 255.f ) );

        __m128 b = _mm_blend_ps( ret, px0, 0x8 );
        _mm_storeu_ps( ptr, b );

        ptr += 4;
        sz--;
    }
}
#  endif

void LinearizePq( float* ptr, int sz )
{
    LinearizePq128( ptr, sz );
}
```
实现了 SIMD 优化的数学函数，包括对数、指数和幂运算，用于替代标准数学库的函数以提高性能。
```c++ 
#if defined __SSE4_1__ && defined __FMA__

static inline __m128 _mm_pow_ps( __m128 x, __m128 y )
{
    return _mm_exp_ps( _mm_mul_ps( y, _mm_log_ps( x ) ) );
}
#endif

```
### SIMD 对数函数
实现原理：使用自然对数的数学性质：log(x) = log(mantissa) + exponent, 将浮点数分解为尾数和指数部分。然后计算对数的多项式逼近。
```c++
static inline __m128 _mm_log_ps( __m128 x )
{
    __m128i e0 = _mm_castps_si128( x );
    // 提取指数部分
    __m128i e1 = _mm_srai_epi32( e0, 23 ); // 右移23位获取指数
    __m128i e2 = _mm_sub_epi32( e1, _mm_set1_epi32( 127 ) ); // 减去偏移量
    // 提取尾数部分并规格化
    __m128i e3 = _mm_and_si128( e0, _mm_set1_epi32( 0x007fffff ) ); // 保留尾数位
    __m128i e4 = _mm_or_si128 ( e3, _mm_set1_epi32( 0x3f800000 ) );
    __m128 e5 = _mm_castsi128_ps( e4 );
    // 使用多项式逼近计算 log(mantissa)
    __m128 f = _mm_sub_ps( e5, _mm_set1_ps( 1.f ) ); // f = mantissa - 1
    __m128 f2 = _mm_mul_ps( f, f );
    __m128 f4 = _mm_mul_ps( f2, f2 );
    __m128 hi = _mm_fmadd_ps( f, _mm_set1_ps( -0.00931049621349f ), _mm_set1_ps(  0.05206469089414f ) );
    __m128 lo = _mm_fmadd_ps( f, _mm_set1_ps(  0.47868480909345f ), _mm_set1_ps( -0.72116591947498f ) );
    hi = _mm_fmadd_ps( f, hi, _mm_set1_ps( -0.13753123777116f ) );
    hi = _mm_fmadd_ps( f, hi, _mm_set1_ps(  0.24187369696082f ) );
    hi = _mm_fmadd_ps( f, hi, _mm_set1_ps( -0.34730547155299f ) );
    lo = _mm_fmadd_ps( f, lo, _mm_set1_ps(  1.442689881667200f ) );
    __m128 r0 = _mm_mul_ps( f, lo );
    __m128 r1 = _mm_fmadd_ps( f4, hi, r0 );
    __m128 r2 = _mm_add_ps( r1, _mm_cvtepi32_ps( e2 ) );
    return r2;
}
```
### SIMD 指数函数
实现原理：利用指数性质：exp(x) = exp(integer_part) * exp(fractional_part), 使用多项式逼近计算 exp(fractional_part)。
```c++
static inline __m128 _mm_exp_ps( __m128 x )
{
    // 分离整数和小数部分
    __m128i mi = _mm_cvtps_epi32( x ); // 整数部分
    __m128  mf = _mm_round_ps( x, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) ); // 四舍五入
    x = _mm_sub_ps( x, mf ); // 小数部分
    // 多项式逼近计算 exp(fractional_part)
    __m128 r = _mm_set1_ps( 1.33336498402e-3f );
    r = _mm_fmadd_ps( x, r, _mm_set1_ps( 9.810352697968e-3f ) );
    r = _mm_fmadd_ps( x, r, _mm_set1_ps( 5.551834031939e-2f ) );
    r = _mm_fmadd_ps( x, r, _mm_set1_ps( 0.2401793301105f ) );
    r = _mm_fmadd_ps( x, r, _mm_set1_ps( 0.693144857883f ) );
    r = _mm_fmadd_ps( x, r, _mm_set1_ps( 1.0f ) );
    // 重新组合结果
    __m128i m0 = _mm_slli_epi32( mi, 23 ); // 左移23位构造指数
    __m128i ri = _mm_castps_si128( r );
    __m128i s = _mm_add_epi32( m0, ri ); // 组合尾数和指数
    __m128 sf = _mm_castsi128_ps( s );
    return sf;
}
```

### 多项式优化
多项式优化通俗解释就是采用数学方法去在一定区域范围内，去逼近我们的exp函数、log函数。多项式计算更加简单。这个可以查看大学高等数学中学到的泰勒级数相关内容。
上面的exp和log函数的多项式逼近中用到的很多奇特的数值，实际上是数学计算推出来的，是固定值，这里可以参考实现[OpenImageIO/fmath](https://github.com/AcademySoftwareFoundation/OpenImageIO/blob/main/src/include/OpenImageIO/fmath.h)。

## simd调试方法

在使用GDB（包括gdb、lldb）的时候，我们希望以更好的阅读方式呈现debug数据内容，也就是gdb的pretty print功能，对于常见的数据结构gdb已经支持，但是simd这种数据并不能打印很好。

simd中128位宽的数据、256、512位的数据，以128为例，可以按照以下数据划分：

64x2 – two lanes of 64-bit integers,
32x4 – four lanes of 32-bit integers,
16x8 – eight lanes of 16-bit integers,
8x16 – sixteen lanes of 8-bit integers.

因此，需要加入pretty print，不然打印出来就是：
```c++
__m128i vPxa = _mm_loadu_si128( (const __m128i *)pPixels );

(gdb) p vPxa
$1 = {8341503235886217471, 8629733612088195327}

(lldb) v vPxa
(__m128i) vPxa = (8341503235886217471, 8629733612088195327)
```
这里[simd-debugging](https://wolf.nereid.pl/posts/simd-debugging/)已经为我们实现了，加入pretty print后的打印：
```c++
(lldb) v vPxa
(__m128i) vPxa = (8341503235886217471, 8629733612088195327)
(lldb) command script import ~/simd.py
(lldb) v vPxa
(__m128i) vPxa = {
  u8x16 = {
    [0] = 255
    [1] = 248
    [2] = 196
    [3] = 115
    [4] = 255
    [5] = 248
    [6] = 194
    [7] = 115
    [8] = 255
    [9] = 248
    [10] = 195
    [11] = 118
    [12] = 255
    [13] = 248
    [14] = 194
    [15] = 119
  }
  i8x16 = {
    [0] = -1
    [1] = -8
    [2] = -60
    [3] = 115
    [4] = -1
    [5] = -8
    [6] = -62
    [7] = 115
    [8] = -1
    [9] = -8
    [10] = -61
    [11] = 118
    [12] = -1
    [13] = -8
    [14] = -62
    [15] = 119
  }
  u16x8 = ([0] = 63743, [1] = 29636, [2] = 63743, [3] = 29634, [4] = 63743, [5] = 30403, [6] = 63743, [7] = 30658)
  i16x8 = ([0] = -1793, [1] = 29636, [2] = -1793, [3] = 29634, [4] = -1793, [5] = 30403, [6] = -1793, [7] = 30658)
  u32x4 = ([0] = 1942288639, [1] = 1942157567, [2] = 1992554751, [3] = 2009266431)
  i32x4 = ([0] = 1942288639, [1] = 1942157567, [2] = 1992554751, [3] = 2009266431)
  u64x2 = ([0] = 8341503235886217471, [1] = 8629733612088195327)
  i64x2 = ([0] = 8341503235886217471, [1] = 8629733612088195327)
}
```
关于gdb pretty print之前也写过一篇文章：[GDB如何优化显示c++ STL数据结构的值](https://zhuanlan.zhihu.com/p/662099267)。、

感谢您的阅读！！