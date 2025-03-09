---
title: LLVM入门-实现一个计算器
date: 2025-03-09
authors: [KenForever1]
categories: 
  - llvm
labels: [llvm]
comments: true
---

## 如何编写一个LLVM Pass

### LLVM介绍

LLVM，全称为低级虚拟机（Low-Level Virtual Machine），是一组模块化和可重用的编译器及工具链技术。由于其灵活的架构，LLVM在多个领域具有广泛的应用。它能够生成机器本地代码，因此被用作主要编程语言（如Rust和Swift）的后端。此外，LLVM还被用于即时编译（JIT编译）以及静态分析、动态分析、着色器编译等多种任务。
<!-- more -->
LLVM的设计目标是提供一个中间表示（IR），以便优化和生成高效的机器代码。它的模块化设计允许开发者根据需要增加或替换组件，从而实现特定的功能或优化。这使得LLVM在学术界和工业界都得到了广泛的应用和支持。

通过不断的发展和演化，LLVM已经成为编译器技术领域的重要组成部分，推动了现代编程语言的发展和性能提升。

### 什么是编译器呢？

编译器是一种软件程序，用于将用高级编程语言编写的源代码翻译成计算机处理器可以执行的形式。一般来说，编译器由三个部分组成：

1. 前端：编译器的前端负责分析源代码，并将其转换为一种中间表示（IR）。这种中间表示通常独立于源编程语言和目标架构，使得可以进行多种优化，并提高代码的可移植性。

2. 中端：编译器的中端对中间表示进行与目标CPU架构无关的优化。这个阶段的重点是进行通用优化，以提高程序的性能和效率。

3. 后端：编译器的后端将优化后的中间表示转换为特定于目标架构的机器代码或汇编代码。后端还负责应用与目标硬件相关的优化，以确保生成的代码能高效地在指定硬件上运行。

通过这三个阶段，编译器能够将人类可读的源代码转化为计算机可以执行的高效机器代码。

### LLVM Pass

在LLVM中，Pass是一种模块化和可重用的组件，旨在对程序的中间表示（IR）进行转换或分析。

根据LLVM的官方文档，Pass主要分为两大类：分析Pass（Analysis Passes）和转换Pass（Transform Passes）。此外，还有一种被称为实用工具Pass（Utility Passes）的类别。

1. 分析Pass（Analysis Passes）：这类Pass用于收集信息和分析代码，而不对代码进行任何修改。它们帮助编译器获取有关程序结构和行为的数据，以便进行进一步的优化和转换。

2. 转换Pass（Transform Passes）：这类Pass用于修改程序的IR，以提高性能、减少代码体积或为进一步的优化做好准备。转换Pass直接对代码进行更改，以实现更高效的执行。

通过这些Pass，LLVM能够实现复杂的代码分析和优化，从而生成高效的机器代码。

### 如何编写一个LLVM Pass

编写LLVM Pass首先需要准备环境，安装上llvm、clang、cmake。安装方法：

```bash
#!/bin/bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 16

```

运行命令会安装llvm-16, clang-16, clang++-16以及相关的工具。头文件安装在：/usr/include/llvm-16/llvm。

LLVM包含两种Pass管理器：旧版Pass管理器（legacy PM）和新版Pass管理器（new PM）。在中端阶段，LLVM使用新版Pass管理器，而在与目标架构相关的后端代码生成阶段，则使用旧版Pass管理器。我们在编写自己的Pass时，可以选择使用旧版或新版管理器。在本文中，我们将使用新版Pass管理器。

新版Pass管理器相较于旧版，通常提供了更好的性能和灵活性，适合用于需要现代优化技术的场合。通过利用新版Pass管理器，我们可以更有效地进行代码分析和优化。


通过下面的例子编译一个简单的Pass，实现将函数名打印出来。

```cpp
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace
{
  // 在LLVM中，所有的Pass都必须继承自CRTP（Curiously Recurring Template Pattern）混入类PassInfoMixin。这种设计模式允许Pass通过模板机制来获得与Pass相关的元信息。
  struct FunctionListerPass : public PassInfoMixin<FunctionListerPass>
  {
    // A pass should have a run method
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM)
    {
      // outs() returns a reference to a raw_fd_ostream for standard output.
      outs() << F.getName() << '\n';
      return PreservedAnalyses::all();
    }
  };

}

PassPluginLibraryInfo getPassPluginInfo()
{
  const auto callback = [](PassBuilder &PB)
  {
    PB.registerPipelineStartEPCallback(
        [&](ModulePassManager &MPM, auto)
        {
          MPM.addPass(createModuleToFunctionPassAdaptor(FunctionListerPass()));
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "name", "0.0.1", callback};
};

// 当驱动程序加载一个插件时，它会调用这个入口点以获取有关该插件的信息，以及如何注册其Pass的信息。
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo()
{
  return getPassPluginInfo();
}

```

将上面的Pass代码编译为一个动态链接库func_lister.so。

```bash
clang-16 -shared -o func_lister.so func_lister.cpp -fPIC -I /usr/inc
lude/llvm-c-16/ -I /usr/include/llvm-16/
```

然后，为了验证我们开发的pass，需要写一个例子test.c。
```cpp
// test.c
void testFunctionOne()
{
}

void testFunctionTwo()
{
}

int main()
{
  return 0;
}
```

编译的时候要制定-O1或者-O2、-O3优化级别，否则不会执行我们的Pass。指定优化级别（例如，用-O1）是重要的，因为它决定了LLVM编译器在编译过程中会应用哪些优化阶段和passes。你会发现不指定优化级别时(默认-O0)，我们的Pass不会被执行。
```bash
clang-16 -O1 -fpass-plugin=./func_lister.so test.c -o test
```
clang-16程序会输出test.c定义的函数名：
```
testFunctionOne
testFunctionTwo
main
```

### 参考

https://llvm.org/docs/WritingAnLLVMNewPMPass.html

https://blog.llvm.org/posts/2021-03-26-the-new-pass-manager/

https://stackoverflow.com/questions/54447985/how-to-automatically-register-and-load-modern-pass-in-clang

https://sh4dy.com/2024/06/29/learning_llvm_01/

## LLVM开发中的一些概念

### Module是什么？

在LLVM中，Module是一个顶层容器，用于封装与整个程序或程序的重要部分相关的所有信息。它是LLVM中间表示（IR）对象的最高级别的容器。每个Module包含以下内容：**全局变量列表**、 **函数列表**、**依赖的库或其他模块的列表**、**符号表**、**目标特性元数据**。

Module在LLVM中扮演着一个关键角色，作为所有相关信息的集成点，用于表示和操作整个程序的结构和行为。它为LLVM的编译和优化过程提供了必要的上下文和组织结构。

### Basic Block是什么？

基本块（Basic Block）是指一组直线执行的指令序列，其中没有分支。这意味着执行从一个单一的入口点开始，按顺序执行到一个单一的出口点，然后继续到下一个基本块。基本块属于某个函数，并且不能在其中间插入跳转，这确保了一旦开始执行，控制流将会顺序执行完该块中的所有指令。

基本块的特点包括：

1. **单一入口和出口**：每个基本块只有一个入口点和一个出口点，没有中途跳转，这保证了指令的线性执行。

2. **连贯性**：一旦程序执行进入基本块，它将按顺序执行该块中的所有指令，直到到达出口点。

3. **函数的组成部分**：基本块是函数中的基本构建单位，函数由一个或多个基本块组成。

4. **领导者指令**：基本块的第一条指令通常被称为“领导者”（leader），因为它是控制流进入该基本块时首先执行的指令。

基本块的结构有助于简化控制流分析和优化，因为它们提供了清晰的执行路径，便于编译器进行优化处理，如常量合并、指令重新排序等。

### Control Flow Graph 是什么？

控制流图（Control Flow Graph，CFG）是一个有向图，其中的节点代表基本块，节点之间的边表示控制流路径，指示执行如何从一个基本块转移到另一个基本块。

控制流图的特点和作用包括：**节点表示基本块**：在CFG中，每个节点对应一个基本块，基本块是程序中一段没有中断的指令序列。**边表示控制流**：节点之间的有向边表示程序在执行时可能的转移路径。也就是说，如果程序在一个基本块结束后可能转移到另一个基本块，那么在CFG中，这两个基本块之间就会有一条边。

CFG是编译器进行程序分析和优化的重要工具。通过分析CFG，编译器可以了解程序的执行流程，从而进行优化，例如死代码消除、循环优化、路径分析等。

![](https://sh4dy.com/images/llvm_learning/llvm_learning_02.png)

![](https://eli.thegreenplace.net/images/2013/09/diamond-cfg.png)

## 编写Pass

前面的Pass注册使用了registerPipelineStartEPCallback，这个小节的例子换成registerPipelineParsingCallback。

```cpp
PassPluginLibraryInfo getPassPluginInfo()
{
    const auto callback = [](PassBuilder &PB)
    {
        PB.registerPipelineParsingCallback(
            [&](StringRef name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>)
            {
                if (name == "run-pass")
                {
                    MPM.addPass(SomePass());
                    return true;
                }
                return false;
            });
    };

    return {LLVM_PLUGIN_API_VERSION, "SomePass", LLVM_VERSION_STRING, callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo()
{
    return getPassPluginInfo();
}
```

可以通过下面的命令，生成包含LLVM IR的.ll 文件，然后使用opt工具执行我们的Pass。

```bash
clang-16 -S -emit-llvm test.c -o test.ll
opt-16 -load-pass-plugin ./lib.so -passes=run-pass -disable-output test.ll
```

```c++
struct SomePass: public PassInfoMixin<SomePass>{
  ...
  static bool isRequired()
  {
    return true;
  }
}

完整可编译代码访问[KenForever1/llvm_snippet](https://github.com/KenForever1/llvm_snippet)

```
### Pass example1: 打印全局变量及其类型
这是一个简单的 LLVM 传递，它会打印出程序中的所有全局变量及其类型。代码遍历所有全局变量，获取它们的名称和类型，并将其打印出来。
```c++
PreservedAnalyses run(Module &M, ModuleAnalysisManager &MPM)
{
  auto globals = M.globals();
  for(auto itr = globals.begin();itr!=globals.end();itr++){
    StringRef varName = itr->getName();
    Type* ty = itr->getType();
    outs()<<"Variable Name: "<<varName<<"\n";
    outs()<<"Variable Type: ";
    ty->print(outs());
    outs()<<"\n";
  }
  return PreservedAnalyses::all();
}
```

完整代码：
```cpp
// main.cpp
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace
{
  struct GlobalVarsPass : public PassInfoMixin<GlobalVarsPass>
  {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &MPM)
    {
      auto globals = M.globals();
      for(auto itr = globals.begin();itr!=globals.end();itr++){
        StringRef varName = itr->getName();
        Type* ty = itr->getType();
        outs()<<"Variable Name: "<<varName<<"\n";
        outs()<<"Variable Type: ";
        ty->print(outs());
        outs()<<"\n";
      }
      return PreservedAnalyses::all();
    }

    static bool isRequired()
    {
      return true;
    }
  };

}

PassPluginLibraryInfo getPassPluginInfo()
{
  const auto callback = [](PassBuilder &PB)
  {
    PB.registerPipelineParsingCallback(
        [&](StringRef name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>)
        {
          if (name == "run-pass")
          {
            MPM.addPass(GlobalVarsPass());
            return true;
          }
          return false;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "GlobalVarsPass", LLVM_VERSION_STRING, callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo()
{
  return getPassPluginInfo();
}
```

test.c例子:

```c++
#include<stdio.h>

int varOne = 1337;
char* name = "sh4dy";
char chr;
float num;

void testFunction(int x){
    if(x==1337){
        puts("nice one");
    }else{
        puts("sad");
    }
}

int main(){
    return 0;
}
```

我们来看一下怎么编译执行：
```bash
#!/bin/bash
clang-16 -shared -o lib.so main.cpp -fPIC
clang-16 -S -emit-llvm test.c -o test.ll
opt-16 -load-pass-plugin ./lib.so -passes=run-pass -disable-output test.ll
```

运行结果：
```bash
Variable Name: varOne
Variable Type: ptr
Variable Name: .str
Variable Type: ptr
Variable Name: name
Variable Type: ptr
Variable Name: .str.1
Variable Type: ptr
Variable Name: .str.2
Variable Type: ptr
Variable Name: chr
Variable Type: ptr
Variable Name: num
Variable Type: ptr
```

### Pass example2: 检测没有使用过的全局变量

实现原理：这段代码遍历所有全局变量并调用use_empty函数。如果该值没有使用者，这个函数将返回 true。

```c++
PreservedAnalyses run(Module &M, ModuleAnalysisManager &MPM)
{
  auto globalVars = M.globals();
  for(GlobalVariable &gvar: globalVars){
    if(gvar.use_empty()){
        outs()<<"Unused global variable: "<<gvar.getName()<<"\n";
    }
  }
  return PreservedAnalyses::all();
}

```
还是上面的执行方法和test.c例子哈，运行结果将会是：
```bash
Unused global variable: varOne
Unused global variable: name
Unused global variable: chr
Unused global variable: num
```

### Pass example3: 打印函数内所有基本块

编写一个用于计算并打印函数内所有基本块的LLVM pass涉及遍历模块中的每个函数，然后检查这些函数中的每个基本块。重要的是要**检查函数是否只是一个声明**，通过isDeclaration函数实现，因为模块通常包含代码中使用的库函数的声明，但它们的完整实现并不在这个模块中。通过检查声明，我们可以避免尝试分析在该模块中没有实际代码的函数。

```c++
// main.cpp部分代码
PreservedAnalyses run(Module &M, ModuleAnalysisManager &MPM)
{
    for (Function &F : M)
    {
        if (!F.isDeclaration())
        {
            int nBlocks = 0;
            outs() << "----------------------------------------------------------------------\n";
            outs() << "Counting and printing basic blocks in the function " << F.getName() << "\n";
            for (BasicBlock &BB : F)
            {
                BB.print(outs());
                outs() << "\n";
                nBlocks++;
            }
            outs() << "Number of basic blocks: " << nBlocks << "\n";
        }
    }
    return PreservedAnalyses::all();
}
```

test.c例子：
```c
#include <stdio.h>

void testFunction(int x)
{
    if (x == 1337)
    {
        puts("nice one");
    }
    else
    {
        puts("sad");
    }
}

void someFunction()
{
    for (int i = 0; i < 10; i++)
    {
        printf("%d\n", i);
    }
}

void anotherFunction()
{
    while (1)
    {
        puts("abcd");
    }
}

int main()
{
    return 0;
}
```

运行结果太长，部门内容如下，会打印每个函数的基本块：
```bash
----------------------------------------------------------------------
Counting and printing basic blocks in the function main

  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  ret i32 0

Number of basic blocks: 1
```

### Pass example4: 检测递归

```c++
PreservedAnalyses run(Module &M, ModuleAnalysisManager &MPM)
{
  for (Function &F : M)
  {
    bool recursionDetected = false;

    for (BasicBlock &BB : F)
    {
      for (Instruction &instr : BB)
      {
        if (instr.getOpcode() == Instruction::Call)
        {
          CallInst *callInstr = dyn_cast<CallInst>(&instr);
          if (callInstr)
          {
            Function *calledFunction = callInstr->getCalledFunction();
            if (calledFunction && calledFunction->getName() == F.getName())
            {
              outs() << "Recursion detected: " << calledFunction->getName() << "\n";
              recursionDetected = true;
              break;
            }
          }
        }
      }
      if (recursionDetected)
        break;
    }
  }
  return PreservedAnalyses::all();
}

```

test.c例子：
```c
int fib(int n)
{
    if (n <= 1)
        return n;
    return fib(n - 1) + fib(n - 2);
}

int testFunction()
{
    return 1000;
}

void anotherFunction()
{
    while (1)
        ;
}

int main()
{
    fib(10);
    return 0;
}
```

运行结果：
```bash
Recursion detected: fib
```

### Pass example5: 对控制流图进行深度优先搜索（DFS）


对于每个函数，我们可以使用F.getEntryBlock()获取第一个基本块，也称为入口块。然后我们调用下面提到的Dfs函数。

```c++
void Dfs(BasicBlock *currentBlock)
{
  static std::unordered_map<BasicBlock *, bool> visited;
  visited[currentBlock] = true;
  currentBlock->print(outs());
  for (BasicBlock *bb : successors(currentBlock))
  {
    if (!visited[bb])
    {
        Dfs(bb);
    }
  }
}

PreservedAnalyses run(Module &M, ModuleAnalysisManager &MPM)
{
  for (Function &F : M)
  {
    if (!F.isDeclaration())
    {
      outs() << "----------------------------------------------------------------\n";
      outs() << "Running DFS for the function " << F.getName() << "\n";
      BasicBlock &entryBlock = F.getEntryBlock();
      Dfs(&entryBlock);
    }
  }
  return PreservedAnalyses::all();
}
```

```c++
#include <stdio.h>

int testFunction()
{
    for (int i = 0; i < 10; i++)
    {
        printf("%d\n", i);
    }
    return 1337;
}

int main()
{
    int x = testFunction();
    if (x > 1000)
    {
        puts("Yay");
    }
    else
    {
        puts("Ne");
    }
    return 0;
}
```

```bash
----------------------------------------------------------------
Running DFS for the function testFunction

  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  br label %2

2:                                                ; preds = %8, %0
  %3 = load i32, ptr %1, align 4
  %4 = icmp slt i32 %3, 10
  br i1 %4, label %5, label %11

5:                                                ; preds = %2
  %6 = load i32, ptr %1, align 4
  %7 = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %6)
  br label %8

8:                                                ; preds = %5
  %9 = load i32, ptr %1, align 4
  %10 = add nsw i32 %9, 1
  store i32 %10, ptr %1, align 4
  br label %2, !llvm.loop !6

11:                                               ; preds = %2
  ret i32 1337
----------------------------------------------------------------
Running DFS for the function main

  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  %3 = call i32 @testFunction()
  store i32 %3, ptr %2, align 4
  %4 = load i32, ptr %2, align 4
  %5 = icmp sgt i32 %4, 1000
  br i1 %5, label %6, label %8

6:                                                ; preds = %0
  %7 = call i32 @puts(ptr noundef @.str.1)
  br label %10

10:                                               ; preds = %8, %6
  ret i32 0

8:                                                ; preds = %0
  %9 = call i32 @puts(ptr noundef @.str.2)
  br label %10
```

在上面的test.c中存在四个basic block，我们可以通过命令生成dot文件，然后使用dot工具生成图片。
```bash
# 安装dot工具
sudo apt install graphviz

# Generate the LLVM IR
clang-16 -S -emit-llvm test.c -o test.ll

# Print Control-Flow Graph to 'dot' file.
opt-16 -dot-cfg -disable-output -enable-new-pm=0 test.ll

# Generate an image from the 'dot' file
dot -Tpng -o img.png .main.dot

```

![](https://sh4dy.com/images/llvm_learning/llvm_learning_03.png)

## 实现一个计算器

### 什么是JIT？
JIT（即时编译器）是一种在程序开始执行后才动态生成机器代码的编译器。与之相对的是AOT（预先编译器），它在程序执行之前将源代码翻译成可执行代码，这些可执行代码存储在编译后的二进制文件中。

具体来说，JIT编译器在运行时对代码进行编译，这意味着它可以在程序执行的过程中优化代码性能，因为它可以利用运行时的信息来进行更有效的优化。这种动态编译的方式允许程序在执行时进行调整，从而提高效率和性能。

相反，AOT编译器在程序运行之前就已经将代码编译好了。这样做的好处是程序启动速度更快，因为可执行代码已经准备就绪。缺点是在编译时不能利用运行时信息来优化代码。

### 什么是ORC

ORC（On-Request Compilation，按需编译）是LLVM JIT API的第三代实现。在ORC之前，有MCJIT，再往前则是Legacy JIT。ORC提供了一种更灵活和模块化的方式来进行即时编译。

与之前的JIT实现相比，ORC的设计目标之一是提高编译过程的灵活性和可扩展性。它允许开发者更精细地控制代码生成和优化过程，并支持更复杂的用例。

在使用ORC JIT API时，开发者可以动态地生成机器代码，这对于需要高性能和灵活性的应用来说非常有用。比如在一个小型计算器项目中，ORC可以用来在运行时即时编译和执行用户输入的计算表达式。通过ORC API，开发者可以编写程序，在需要时动态生成和执行高效的机器代码。

## LLVM中的JIT

LLVM的JIT（即时编译）功能是一种强大的机制，可以在程序运行时动态生成和优化机器代码。这对于需要高性能和灵活性的应用程序非常有用，比如解释器、动态语言运行时或特定场景的优化代码。以下是LLVM JIT的一些关键点和如何使用它的基本步骤：

+ **LLVM IR**：LLVM的中间表示，JIT编译器会将这种中间表示转换为机器代码。LLVM IR是平台无关的，并且易于分析和优化。

+ **ExecutionEngine**：这是LLVM提供的用于执行JIT编译代码的核心组件。ExecutionEngine负责将LLVM IR转换为本地机器代码，并在目标平台上执行。

+ **MCJIT和ORC**：LLVM中有两种主要的JIT引擎：
   - **MCJIT**：一种较旧的JIT实现，提供了基本的JIT编译功能。
   - **ORC（On Request Compilation）**：一种更现代和模块化的JIT实现，提供了更灵活和强大的功能，比如更好的代码分离和内存管理。

### 使用LLVM JIT的基本步骤

1. **创建LLVM模块和IR**：首先，需要构建一个LLVM模块，并使用IR Builder生成LLVM IR。这一步通常涉及定义函数、基本块和指令。

2. **初始化JIT引擎**：选择并初始化一个JIT引擎（MCJIT或ORC）。这包括创建ExecutionEngine和配置目标机器。

3. **添加模块到引擎**：将LLVM模块添加到JIT引擎中。引擎会负责将模块中的IR转换为机器代码。

4. **查找和执行函数**：通过引擎查找已编译的函数的入口点，然后通过函数指针调用它们。

### 代码实现

定义了add, sub, mul, xor四个指令，从txt文件中读取，每一行都是一个语句。

```
add val1, val2  
sub val1, val2  
mul val1, val2  
xor val1, val2  
```
定义一个结构体来保存指令信息，

```c++
struct Instruction {
    std::string name; // add，sub，mul，xor
    int64_t val1;
    int64_t val2;

    Instruction(const std::string &name, int64_t val1, int64_t val2) 
        : name(name), val1(val1), val2(val2) {}
};
```

从文件读取所有的指令，保存到vector中。
```c++
std::vector<std::unique_ptr<Instruction>> GetInstructions(const std::string &file_name) {
    std::ifstream ifile(file_name);
    std::string instruction_line;
    std::vector<std::unique_ptr<Instruction>> instructions;

    if (!ifile.is_open()) {
        fatal_error("Failed to open file: " + file_name);
    }

    while (std::getline(ifile, instruction_line)) {
        std::istringstream stream(instruction_line);
        std::string instruction_type;
        int64_t val1, val2;
        char comma;

        if (stream >> instruction_type >> val1 >> comma >> val2) {
            instructions.push_back(std::make_unique<Instruction>(instruction_type, val1, val2));
        } else {
            fatal_error("Invalid instruction format: " + instruction_line);
        }
    }
    return instructions;
}
```

现在我们已经覆盖了基础知识，可以开始处理特定于LLVM的任务。在JIT（即时编译）我们的代码之前，我们需要为所有希望JIT编译的函数生成相应的LLVM IR（中间表示）。要生成LLVM IR，首先需要创建一个LLVM上下文、一个LLVM模块和一个IR生成器。

- 上下文（Context）：上下文是一个容器，用于拥有和管理LLVM特定的核心数据结构。

- 模块（Module）：LLVM模块是一个顶级容器，表示一个编译单元，包含函数、全局变量以及其他程序元素，比如该模块依赖的库（或其他模块）列表、符号表等。

- 基本块（Basic Block）：基本块是一组线性顺序的指令序列，没有分支，意味着执行从一个单一入口点开始，按顺序进行到一个单一出口点，随后继续到下一个基本块。基本块属于函数，不能在中间有跳转，确保一旦开始执行，就会按顺序执行完该块内的所有指令。基本块的第一条指令被称为领导者（leader）。

我们的目标是为每条指令生成一个LLVM IR形式的单独函数。例如，与add指令对应的函数将如下所示：

```c++
define i64 @add(i64 %0, i64 %1) {
entry:
  %2 = add i64 %0, %1
  ret i64 %2
}
```

写一个函数来为add, sun, mul, xor生成LLVM IR，
```c++
void AddFunctionsToIR(llvm::LLVMContext &ctx, llvm::Module *module, const std::string &function_name) {
    auto int64_type = llvm::Type::getInt64Ty(ctx);
    std::vector<llvm::Type *> params(2, int64_type);
    llvm::IRBuilder<> ir_builder(ctx);

    llvm::FunctionType *function_type = llvm::FunctionType::get(int64_type, params, false);
    llvm::Function *func = llvm::Function::Create(function_type, llvm::Function::ExternalLinkage, function_name, module);

    // Create the entry block for the function
    llvm::BasicBlock *basic_block = llvm::BasicBlock::Create(ctx, "entry", func);
   
    // Append instructions to the basic block
    ir_builder.SetInsertPoint(basic_block);

    auto args = func->args();
    auto arg_iter = args.begin();
    llvm::Argument *arg1 = arg_iter++;
    llvm::Argument *arg2 = arg_iter;

    llvm::Value *result = nullptr;

    if (function_name == "add") {
        result = ir_builder.CreateAdd(arg1, arg2);
    } else if (function_name == "sub") {
        result = ir_builder.CreateSub(arg1, arg2);
    } else if (function_name == "mul") {
        result = ir_builder.CreateMul(arg1, arg2);
    } else if (function_name == "xor") {
        result = ir_builder.CreateXor(arg1, arg2);
    } else {
        fatal_error("Invalid function name: " + function_name);
    }

    // return the value
    ir_builder.CreateRet(result);
}

```
这段代码为我们的每条指令生成函数。每个函数接受两个参数，并根据操作返回一个值。现在我们已经创建了这个函数，让我们继续编写主函数的代码。

调用两个关键的函数进行初始化：
```c++
/*
Initialize the native target corresponding to the host
*/
llvm::InitializeNativeTarget();


/* Calling this function is also necessary for code generation.
 It sets up the assembly printer for the native host architecture.
*/
llvm::InitializeNativeTargetAsmPrinter();
```

创建ctx和module。
```c++
llvm::LLVMContext ctx;
auto module = std::make_unique<llvm::Module>("my_module", ctx);

```

调用上面的AddFunctionsToIR函数，为add, sub, mul, xor生成LLVM IR。

```c++
AddFunctionsToIR(ctx, module.get(), "add");
AddFunctionsToIR(ctx, module.get(), "sub");
AddFunctionsToIR(ctx, module.get(), "mul");
AddFunctionsToIR(ctx, module.get(), "xor");
```

我们可以创建一个 LLJIT 生成器的实例。LLJIT 是 LLVM 的 ORC（按需编译）JIT 引擎的一部分，它为即时编译提供了一个现代、灵活和模块化的基础设施，是 MCJIT 的合适替代品。

```c++
auto jit_builder = llvm::orc::LLJITBuilder();
auto jit = jit_builder.create();
```

接下来，就是将我们的模块添加到主JITDylib（一个JITDylib代表了一个JIT的动态库）。

```c++
if (auto err = jit->get()->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::make_unique<llvm::LLVMContext>()))) {
    fatal_error("Failed to add IR module for JIT compilation: " + llvm::toString(std::move(err)));
}

```

```c++
llvm::orc::ExecutorAddr GetExecutorAddr(llvm::orc::LLJIT &jit, const std::string &function_name) {
    auto sym = jit.lookup(function_name).get();
    if (!sym) {
        fatal_error("Function not found in JIT: " + function_name);
    }
    return sym;
}

```
解析code.txt文件，也就是每一行都是一条指令。根据指令的名称，比如add去获取llvm::orc::ExecutorAddr。通过强制转换拿到fn，执行。
```c++
// main
auto instructions = GetInstructions("code.txt");
std::unordered_map<std::string, llvm::orc::ExecutorAddr> fn_symbols;

for (const auto &instruction : instructions) {
    if (fn_symbols.find(instruction->name) == fn_symbols.end()) {
        fn_symbols[instruction->name] = GetExecutorAddr(*jit->get(), instruction->name);
    }

    auto *fn = reinterpret_cast<int64_t (*)(int64_t, int64_t)>(fn_symbols[instruction->name].getValue());
    int64_t value = fn(instruction->val1, instruction->val2);
    std::cout << value << std::endl;
}
```

为了避免多次获取函数地址，我们可以将函数地址存储在一个哈希表中。这样，每次需要执行指令时，我们都可以直接从哈希表中获取函数地址，而不需要每次都通过 JIT 查找。

```c++
std::unordered_map<std::string, llvm::orc::ExecutorAddr> fn_symbols;
```

完整的代码如下：

```c++
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <memory>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

struct Instruction {
    std::string name;
    int64_t val1;
    int64_t val2;

    Instruction(const std::string &name, int64_t val1, int64_t val2) 
        : name(name), val1(val1), val2(val2) {}
};

void fatal_error(const std::string &message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

std::vector<std::unique_ptr<Instruction>> GetInstructions(const std::string &file_name) {
    std::ifstream ifile(file_name);
    std::string instruction_line;
    std::vector<std::unique_ptr<Instruction>> instructions;

    if (!ifile.is_open()) {
        fatal_error("Failed to open file: " + file_name);
    }

    while (std::getline(ifile, instruction_line)) {
        std::istringstream stream(instruction_line);
        std::string instruction_type;
        int64_t val1, val2;
        char comma;

        if (stream >> instruction_type >> val1 >> comma >> val2) {
            instructions.push_back(std::make_unique<Instruction>(instruction_type, val1, val2));
        } else {
            fatal_error("Invalid instruction format: " + instruction_line);
        }
    }
    return instructions;
}

void AddFunctionsToIR(llvm::LLVMContext &ctx, llvm::Module *module, const std::string &function_name) {
    auto int64_type = llvm::Type::getInt64Ty(ctx);
    std::vector<llvm::Type *> params(2, int64_type);
    llvm::IRBuilder<> ir_builder(ctx);

    llvm::FunctionType *function_type = llvm::FunctionType::get(int64_type, params, false);
    llvm::Function *func = llvm::Function::Create(function_type, llvm::Function::ExternalLinkage, function_name, module);

    llvm::BasicBlock *basic_block = llvm::BasicBlock::Create(ctx, "entry", func);

    // Append instructions to the basic block
    ir_builder.SetInsertPoint(basic_block);

    auto args = func->args();
    auto arg_iter = args.begin();
    llvm::Argument *arg1 = arg_iter++;
    llvm::Argument *arg2 = arg_iter;

    llvm::Value *result = nullptr;

    if (function_name == "add") {
        result = ir_builder.CreateAdd(arg1, arg2);
    } else if (function_name == "sub") {
        result = ir_builder.CreateSub(arg1, arg2);
    } else if (function_name == "mul") {
        result = ir_builder.CreateMul(arg1, arg2);
    } else if (function_name == "xor") {
        result = ir_builder.CreateXor(arg1, arg2);
    } else {
        fatal_error("Invalid function name: " + function_name);
    }

    ir_builder.CreateRet(result);
 }

llvm::orc::ExecutorAddr GetExecutorAddr(llvm::orc::LLJIT &jit, const std::string &function_name) {
    auto sym = jit.lookup(function_name).get();
    if (!sym) {
        fatal_error("Function not found in JIT: " + function_name);
    }
    return sym;
}

int main() {
    llvm::LLVMContext ctx;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    auto module = std::make_unique<llvm::Module>("neko_module", ctx);

    AddFunctionsToIR(ctx, module.get(), "add");
    AddFunctionsToIR(ctx, module.get(), "sub");
    AddFunctionsToIR(ctx, module.get(), "mul");
    AddFunctionsToIR(ctx, module.get(), "xor");

    auto jit_builder = llvm::orc::LLJITBuilder();
    auto jit = jit_builder.create();
    if (!jit) {
        fatal_error("Failed to create JIT: " + llvm::toString(jit.takeError()));
    }

    if (auto err = jit->get()->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::make_unique<llvm::LLVMContext>()))) {
        fatal_error("Failed to add IR module for JIT compilation: " + llvm::toString(std::move(err)));
    }

    auto instructions = GetInstructions("code.txt");
    std::unordered_map<std::string, llvm::orc::ExecutorAddr> fn_symbols;

    for (const auto &instruction : instructions) {
        if (fn_symbols.find(instruction->name) == fn_symbols.end()) {
            fn_symbols[instruction->name] = GetExecutorAddr(*jit->get(), instruction->name);
        }

        auto *fn = reinterpret_cast<int64_t (*)(int64_t, int64_t)>(fn_symbols[instruction->name].getValue());
        int64_t value = fn(instruction->val1, instruction->val2);
        std::cout << value << std::endl;
    }

    return 0;
}

```

我们可以使用cmake来编译程序, CMakeLists.txt内容如下：
```c++
cmake_minimum_required(VERSION 3.13)
project(main)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM 16 REQUIRED CONFIG)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

llvm_map_components_to_libnames(
    llvm_libs
    core
    orcjit
    native
)

target_link_libraries(${PROJECT_NAME} ${llvm_libs})
```

```bash

cmake -B build -S .

cmake --build build
```

创建一个code.txt文件用来测试：
```c++
add 1,2
sub 10,5
mul 10,20
xor 5,5
add 5,10
xor 10,5
```
运行结果：
```bash
3
5
200
0
15
15
```

那些经过即时编译（JIT）的函数的代码。这些代码存储在一个区域中，一旦即时编译器将代码写入该区域，该区域的权限随后会被设置为可读可执行。

## 参考

+ https://sh4dy.com/2024/11/24/learning_llvm_03/

+ https://llvm.org/docs/ORCv2.html

+ https://liuyehcf.github.io/2023/07/10/LLVM-JIT/