---
comments: true
---
# MLIR初遇--Toy2

[TOC]
## 初遇
+ AI编译器和AI部署框架的区别？
+ MLIR和tvm的区别？
## MLIR的作用
初次接触MLIR是对模型进行量化操作的工具，通过MLIR多层级的逐步Lowering，然后可以对算子进行融合等优化操作。
AI编译器最常用的是使用MLIR方式和tvm方式，tvm方式采用类似遗传算法等在一个空间进行自动搜索，得到优化结果。MLIR提供了一个基础的框架，可以通过将专家知识编写成各种Pass，逐层优化的IR，最终部署到各种硬件设备上。

比如：
+ [sophgo tpu-mlir](https://github.com/sophgo/tpu-mlir) 算能科技的编译器，对模型进行量化，部署到TPU上。
+ [triton](https://github.com/triton-lang/triton)，有两个知名的项目都叫做triton，一个是英伟达出的服务端部署框架[tritonserver](https://github.com/triton-inference-server/server)，一个是openai根据mlir的一个编程语言。

## 编译llvm-project项目
通过llvm-project项目中的toy example对mlir进行了源码学习和调试，一起一窥究竟。
编译命令：
```bash
cmake -G Ninja ../llvm    -DLLVM_ENABLE_PROJECTS=mlir    -DLLVM_BUILD_EXAMPLES=ON    -DLLVM_TARGETS_TO_BUILD="X86"    -DCMAKE_BUILD_TYPE=Release    -DLLVM_ENABLE_ASSERTIONS=ON  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON

# cmake -G Ninja ../llvm    -DLLVM_ENABLE_PROJECTS=mlir    -DLLVM_BUILD_EXAMPLES=ON    -DLLVM_TARGETS_TO_BUILD="X86"    -DCMAKE_BUILD_TYPE=Debug    -DLLVM_ENABLE_ASSERTIONS=ON  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON

cmake --build . --target check-mlir
```

## 自定义Op

根据toy2的文档描述，我们知道定义一个Dialect和相关的Op，可以有两种方式，第一种是采用他专门的描述语言编写td文件，通过mlir-tblgen命令可以生成头文件声明和函数定义。第二种是手动编写cpp文件。

在toy2的
```bash
.
├── CMakeLists.txt
├── include
│   ├── CMakeLists.txt
│   └── toy
│       ├── AST.h
│       ├── CMakeLists.txt
│       ├── Dialect.h
│       ├── Lexer.h
│       ├── MLIRGen.h
│       ├── Ops.td
│       └── Parser.h
├── mlir
│   ├── Dialect.cpp
│   └── MLIRGen.cpp
├── parser
│   └── AST.cpp
└── toyc.cpp

4 directories, 13 files
```

首先对文件相关内容进行一个简单的介绍：
+ AST和Parser相关的都是对自定义的Toy语言（类似python函数定义语法def，只支持double类型）进行词法语法分析的内容，如果只是学习MLIR编译可以先跳过
+ Dialect.h导入了td文件生成的头文件定义和mlir相关头文件
+ Ops.td文件通过td特有的这种领域语言描述了ToyDialect和ToyOp，通过命令生成代码，避免了大量的编写cpp的工作
+ Dialect.cpp中有部分手写cpp实现的Op等
+ toyc.cpp实现了命令行读取，DumpAst，DumpMlir的功能，原生的MLIR肯定不能解析toy定义的内容，因为原生MLIR提供的默认Dialect根本都不认识Toy的语法，那么在toyc.cpp中对ToyDialect进行了注册
  
toy.cpp中注册toyDialect的逻辑：
```cpp
// toy.cpp
int dumpMLIR() {
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();
    ......
}
```
在inlude/toy/Ops.td文件中，定义了ToyDialect和相关的Op，比如PrintOp。
在mlir/Dialect.cpp中手动实现了部分Op，比如ConstantOp的Parse和Print函数。相对比PrintOp的Parse函数和Print函数是td文件定义，然后通过mlir-tblgen命令生成的。

下面通过代码的小改动来了解这个过程，你将会了解到如下内容：
+ 如何使用命令生成Dialect的声明和定义？
+ 如何使用命令生成Op的声明和定义，如何过滤只生成某个Op的声明和定义？
+ 如何为PrintOp手写Parse函数和Print函数？
+ 如何将编译代码，将改动应用到生成的mlir文件中？

### 正常编译toy示例流程

先看一下toy例子定义，在mlir/test/Examples/Toy/Ch2/codegen.toy文件中：

```cpp
// codegen.toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```
可以看到定义了两个函数multiply_transpose和main主函数，在main函数中调用了print函数，也就是printOp。
第一步，编译toy文件
```bash
#!/bin/bash
mlir_src_root=$(pwd)/mlir
build_root=$(pwd)/build

${build_root}/bin/toyc-ch2 ${mlir_src_root}/test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo
```

编译后生成的内容：

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":4:1)) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":5:25)
    toy.return %2 : tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":5:3)
  } loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":4:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":12:11)
    toy.print %5 : tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":13:3)
    toy.return loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":8:1)
  } loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```
上面就是编译生成的mlir的内容，我们先关注print调用的部分代码：
```mlir
toy.print %5 : tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":13:3)
```
打印中的内容和格式，在td文件中进行了定义：
```
def PrintOp : Toy_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins F64Tensor:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```
assemblyFormat就是输出的定义，包括了$input以及输入类型等。
```
let assemblyFormat = "$input attr-dict `:` type($input)";
```
第二步，对PrintOp的声明和定义进行生成。
```
${build_root}/bin/mlir-tblgen -gen-op-decls ${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td --op-include-regex "print" -I ${mlir_src_root}/include/ 
```
**op-include-regex**对op进行了过滤，指定只生成printOp的声明。
执行结果：
```cpp
{{
......
class PrintOp : public ::mlir::Op<PrintOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PrintOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PrintOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.print");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
public:
};
} // namespace toy
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::PrintOp)
```

可以看到声明中有parse和print函数。下面再生成函数定义，命令如下：
```cpp
//===----------------------------------------------------------------------===//
// ::mlir::toy::PrintOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
PrintOpAdaptor::PrintOpAdaptor(PrintOp op) : PrintOpGenericAdaptor(op->getOperands(), op) {}
.......

::mlir::ParseResult PrintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PrintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace toy
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::PrintOp)
```
可以看到生成的parse函数和print函数的定义，print函数很好理解，获取内容，如getInput()，然后调用
_odsPrinter << 打印内容，就像cout一样。

### 手写PrintOp的Print和Parse方法cpp内容

刚刚生成了Print函数和Parse函数的定义，我们如果要手动改写cpp怎么办呢？聪明如你想到了，把生成的两个函数粘贴到Dialect.cpp中。
那么怎么避免自动生成这两个函数呢？参考ConstantOp操作，改写了td文件中PrintOp的定义，注释了assemblyFormat，添加hasCustomAssemblyFormat。意思是，不指定Format，然后自定义Format。

```
// Ops.td
def PrintOp : Toy_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins F64Tensor:$input);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

粘贴Print函数和Parse函数到Dialect.cpp中。
```cpp
void PrintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  _odsPrinter << "hello_change";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}
```

由于改动了cpp源码，需要先编译toy-ch2
```bash
cmake --build . --target toyc-ch2
```

重新执行：
```bash
#!/bin/bash
mlir_src_root=$(pwd)/mlir
build_root=$(pwd)/build

${build_root}/bin/toyc-ch2 ${mlir_src_root}/test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo
```
生成的mlir内容如下，可以看到其它内容没有发生变化，但是增加了我们新添加的"hello_change"内容。
```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":4:1)) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":5:25)
    toy.return %2 : tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":5:3)
  } loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":4:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":12:11)
    toy.print %5hello_change : tensor<*xf64> loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":13:3)
    toy.return loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":8:1)
  } loc("/home/ken/Codes/mlir_about/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```
## 参考
https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/