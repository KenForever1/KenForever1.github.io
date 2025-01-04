合理运用 L2 缓存能够提速运算。A100 的 L2 缓存能够设置至多 40MB 的持续化数据 (persistent data)，能够拉升算子 kernel 的带宽和性能。Flash attention 的思路就是尽可能地利用 L2 缓存，减少 HBM 的数据读写时间。

在图形处理器（GPU）中，内存技术扮演着至关重要的角色，不同类型的内存（DRAM、HBM、SRAM）各有其特定的用途和优势。以下是对这些内存在GPU中的介绍：

### 0.1 DRAM（动态随机存取存储器）

**概述：**  
DRAM是一种常见的半导体存储器，广泛用于电脑、服务器以及GPU等设备中。它通过存储电荷在电容器中表示数据的“0”或“1”。

**在GPU中的应用：**

- **显存（VRAM）：** GPU使用DRAM作为显存，用于存储图形数据、纹理、帧缓冲区等。
- **容量优势：** DRAM具有较高的存储密度，能够提供大量的内存容量，适合需要处理复杂图形和高分辨率的应用场景。

**特点：**

- **高容量：** DRAM可以在较小的物理空间内存储大量数据。
- **相对较低的成本：** 与其他类型的内存（如SRAM、HBM）相比，DRAM的单位存储成本较低。
- **较高的延迟和较低的带宽：** 虽然容量大，但DRAM的访问速度较慢，带宽也不及HBM，可能成为高性能GPU的瓶颈。

### 0.2 HBM（高带宽内存）

**概述：**  
HBM是一种3D封装的内存技术，旨在提供比传统GDDR（图形双倍数据速率）内存更高的带宽和更低的功耗。它通过将多层内存堆叠并与GPU通过宽接口（通常为堆叠的硅通孔，TSV）连接，实现高效的数据传输。

**在GPU中的应用：**

- **高性能显卡：** HBM主要用于高端显卡和专业GPU，如用于深度学习、科学计算等需要巨大内存带宽的应用。
- **集成方案：** 由于其紧凑的封装，HBM适合用于需要节省空间的集成设计，如一些移动设备和高性能计算加速器。

**特点：**

- **高带宽：** HBM能够提供极高的数据传输速率，通常比传统GDDR内存高出数倍，极大地提升了GPU的并行处理能力。
- **低功耗：** HBM通过短距离高密度的连接，减少了延迟和能耗，提高能效比。
- **紧凑封装：** 3D堆叠设计使得内存占用空间更小，有助于设计更加紧凑和高效的GPU。

### 0.3 SRAM（静态随机存取存储器）

**概述：**  
SRAM是一种快速且稳定的存储器，通过使用多个晶体管保持一个位的状态，不需要周期性刷新电荷，因此称为“静态”。它相比DRAM具有更快的访问速度，但存储密度较低且成本较高。

**在GPU中的应用：**

- **缓存层级：** SRAM主要用于GPU内部的缓存系统，如L1缓存、L2缓存等，用于加速数据访问，提高整体性能。
- **寄存器文件：** GPU的计算单元中使用SRAM作为寄存器文件，以实现快速的数据读写操作。
- **片上资源：** 由于SRAM速度快，适合作为片上高速缓存，减少与主显存（DRAM/HBM）的数据交换延迟。

**特点：**

- **高速访问：** SRAM具有极低的延迟和高数据传输速率，是缓存设计的理想选择。
- **较低的密度和较高的成本：** 相比DRAM，SRAM需要更多的晶体管来存储每一位数据，因此存储密度较低，成本较高。
- **稳定性：** 不需要刷新周期，数据保持更加稳定，适合用于高速缓存和实时计算需求。

### 0.4 总结

在GPU设计中，DRAM、HBM和SRAM各自发挥着不同的作用：

- **DRAM** 提供大量的显存容量，适用于存储复杂图形和大规模数据，但带宽和延迟方面存在限制。
- **HBM** 通过高带宽和低功耗的特性，提升了GPU的整体性能，特别适用于高端和专业应用场景。
- **SRAM** 则作为缓存和寄存器文件，提供高速的数据访问，减少延迟，提升计算效率。

通过合理结合这三种内存技术，现代GPU能够在性能、容量和功耗之间实现平衡，满足各种图形处理和计算需求。

了解GPU的硬件知识是相当重要的，比如在进行TP并行时，需要将数据分散到多个设备进行计算，因此要求GPU之间是互联的。
https://zhuanlan.zhihu.com/p/603908668

GPU的连接方式可能是全连接（通过NVLINK连接，相比PCIE更高速），可能是部分连接（部分NVLINK，部分PCIE）。

矩阵乘法，比如A@B，可以将A按照行分成多个矩阵，也可以将B按照列分成多个矩阵。然后分别计算，再Gather数据合并成最终结果。

使用 PyTorch 在多个设备（如 GPU）上进行矩阵计算，可以通过模型并行（而不是数据并行）的方式实现。这涉及将矩阵拆分到多个设备上进行部分计算，然后汇总结果。本示例将展示如何使用 PyTorch 在多 GPU 上拆分矩阵乘法并进行合并操作。

假设我们有两大矩阵 A 和 B，希望将它们的乘法结果拆分到两个 GPU 上进行计算：
```python
import torch

import time

# 检查是否有多个 GPU 可用
if torch.cuda.device_count() < 2:
raise RuntimeError("This script requires at least 2 GPUs.")

# 设置设备
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')

# 创建两个大的矩阵

n = 2000 # 为了更好的展示效果，选择较大的尺寸

A = torch.rand((n, n), device=device0)

B = torch.rand((n, n), device=device1)


# 将 B 分块到不同的设备上

B1 = B[:, :n//2].to(device0)

B2 = B[:, n//2:].to(device1)

# 为结果矩阵保留空间

C1 = torch.zeros((n, n//2), device=device0)

C2 = torch.zeros((n, n//2), device=device1)

# 记录开始时间

start_time = time.time()

# 分块矩阵乘法，计算 A * B1 和 A * B2 分别在两块 GPU 上

C1 = torch.matmul(A, B1)

C2 = torch.matmul(A.to(device1), B2)

# 汇总结果到单一设备上

C_full = torch.cat((C1, C2.to(device0)), dim=1)

# 记录结束时间

torch.cuda.synchronize() # 确保所有GPU计算任务完成

end_time = time.time()

print(f"Multi-GPU computation time: {end_time - start_time:.6f} seconds")
```


分布式并行计算中，all-gather和all-reduce的区别？
图： Guide_Laboratory_NCCL_MPI_Collective_Operations_english-2021@muriloboratto.pdf

```bash
假设有三个节点 A, B, C： - 初始状态：A=[1], B=[2], C=[3] - All-Gather 后：A=[1,2,3], B=[1,2,3], C=[1,2,3]
```

```bash
假设有三个节点 A, B, C：
- 初始状态：A=[1], B=[2], C=[3]
- All-Reduce（加法）后：A=[6], B=[6], C=[6] (因为 1 + 2 + 3 = 6)
```

参考：
https://medium.com/@sachinkalsi/flashattention-understanding-gpu-architecture-part-1-0a8a9a0bb725

https://zhuanlan.zhihu.com/p/462191421

https://www.cnblogs.com/ArsenalfanInECNU/p/18021724

https://zhuanlan.zhihu.com/p/638468472

https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html

https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
