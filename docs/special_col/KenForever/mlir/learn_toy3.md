# MLIR 系列--Toy3 Rewrite Pass
[TOC]

上一篇：[MLIR初遇--Toy2](https://zhuanlan.zhihu.com/p/711422122)
> https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/

通过模式匹配完成Op的优化，比如对一个矩阵做两次transpose转置操作，等于原始矩阵。那么就可以优化为不做这个操作，节约两次操作。

## 两种机制
+ cpp继承OpRewritePattern实现
+ 通过DRR描述

## cpp Rewrite Transpose操作
toy描述如下：
```
def transpose_transpose(x) {
  return transpose(transpose(x));
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b = transpose_transpose(a);
  print(b);
}
```

优化前生成的mlir。
```bash
${build_root}/bin/toyc-ch3 ${mlir_src_root}/test/Examples/Toy/Ch3/transpose_transpose.toy -emit=mlir
```

```mlir
module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    toy.return %1 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
    %2 = toy.generic_call @transpose_transpose(%1) : (tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %2 : tensor<*xf64>
    toy.return
  }
}
```

```bash
${build_root}/bin/toyc-ch3 ${mlir_src_root}/test/Examples/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt
```
优化后生成的mlir。
```mlir
module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    toy.return %arg0 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.generic_call @transpose_transpose(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %1 : tensor<*xf64>
    toy.return
  }
}
```

我们在matchAndRewrite函数中加入打印信息：
```cpp
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    std::cout << "call matchAndRewrite transpose op "<< op.getOperationName().str() << "\n";
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();
    op.emitWarning() << "arrive here" << "\n";
    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};
```

```
def transpose_transpose(x) {
  return transpose1(transpose2(x));
}
```

通过打印，可以看到匹配了两次，匹配到了transpose(transpose())形式，对其进行消除。
```
call matchAndRewrite transpose op toy.transpose
call matchAndRewrite transpose op toy.transpose
arrive here
```
debug模式下打印追踪情况:
```bash
//===-------------------------------------------===//
/xxx/llvm-project/mlir/test/Examples/Toy/Ch3/transpose_transpose.toy:5:20: warning: index : 1 call matchAndRewrite transpose op toy.transpose

  return transpose(transpose(x));
                   ^
/xxx/llvm-project/mlir/test/Examples/Toy/Ch3/transpose_transpose.toy:5:20: note: see current operation: %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
/xxx/llvm-project/mlir/test/Examples/Toy/Ch3/transpose_transpose.toy:5:10: warning: index : 2 call matchAndRewrite transpose op toy.transpose

  return transpose(transpose(x));
         ^
/xxx/llvm-project/mlir/test/Examples/Toy/Ch3/transpose_transpose.toy:5:10: note: see current operation: %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
/xxx/llvm-project/mlir/test/Examples/Toy/Ch3/transpose_transpose.toy:5:10: warning: arrive here

  return transpose(transpose(x));
         ^
/xxx/llvm-project/mlir/test/Examples/Toy/Ch3/transpose_transpose.toy:5:10: note: see current operation: %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
```

## 注册规范化（Canonicalizer）机制

在上面我通过-opt打开了Canonicalizer的优化选项，进行了模式匹配应用我们的优化规则。对应的实现在toyc.cpp文件中如下，
```cpp
// "toyc.cpp"
if (enableOpt) {
mlir::PassManager pm(module.get()->getName());
// Apply any generic pass manager command line options and run the pipeline.
if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

// Add a run of the canonicalizer to optimize the mlir module.
pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
if (mlir::failed(pm.run(*module)))
    return 4;
}
```

在ToyCombine.cpp中，将Rewriter进行了注册，
```cpp
// Register our patterns as "canonicalization" patterns on the TransposeOp so
// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
```

## DRR方式Rewrite Reshape

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>
    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
    %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
    toy.print %3 : tensor<2x1xf64>
    toy.return
  }
}
```
主要实现了两种优化：
+ Reshape(x) -> x 优化为 x，即Reshape的source和dst一样时，省略Reshape操作
+ Reshape(constant(x)) 优化为x，即对一个constant做Reshape操作，省略Reshape操作
```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

DDR方式描述的内容在mlir/examples/toy/Ch3/mlir/ToyCombine.td中，相对cpp描述的方式更加简单。
```
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;

def ReshapeConstant :
  NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;

def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

DDR这种声明的方式不清晰，为了看生成的cpp内容，可以通过mlir-tblgen命令, --gen-rewriters会生成rewriter对应的cpp代码：
```bash
${build_root}/bin/mlir-tblgen --gen-rewriters  ${mlir_src_root}/examples/toy/Ch3/mlir/ToyCombine.td -I ${mlir_src_root}/include/  -I ${mlir_src_root}/examples/toy/Ch3/include
```

生成的cpp内容部分如下，以ReshapeReshapeOptPattern为例，包括了解析操作，拿到两个父子ReshapeOp和参数，执行relapceOp操作。最后对ReshapeReshapeOptPattern进行注册。
```cpp
struct ReshapeReshapeOptPattern : public ::mlir::RewritePattern {
  ReshapeReshapeOptPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("toy.reshape", 2, context, {"toy.reshape"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range arg(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::toy::ReshapeOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::toy::ReshapeOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::toy::ReshapeOp type";
        });
      }
      arg = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::toy::ReshapeOp tblgen_ReshapeOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ReshapeOp_0 = rewriter.create<::mlir::toy::ReshapeOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ReshapeOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<FoldConstantReshapeOptPattern>(patterns.getContext());
  patterns.add<RedundantReshapeOptPattern>(patterns.getContext());
  patterns.add<ReshapeReshapeOptPattern>(patterns.getContext());
}
```

## debug调试匹配过程

在模式匹配过程中，如何了解模式匹配的过程呢？哪些模式成功匹配了，哪些模式匹配失败了。
在运行命令的时候通过：-debug参数。例如：
```
${build_root}/bin/toyc-ch3 ${mlir_src_root}/test/Examples/Toy/Ch3/trivial_reshape.toy -emit=mlir -debug -opt
```
会有如下打印信息，这是在匹配第一个Reshape操作，匹配reshape（constant）模式成功，生成了mlir：
```bash

  * Pattern (anonymous namespace)::ReshapeReshapeOptPattern : 'toy.reshape -> (toy.reshape)' {
Trying to match "(anonymous namespace)::ReshapeReshapeOptPattern"
    ** Match Failure : castedOp1 is not ::mlir::toy::ReshapeOp type
"(anonymous namespace)::ReshapeReshapeOptPattern" result 0
  } -> failure : pattern failed to match

  * Pattern (anonymous namespace)::FoldConstantReshapeOptPattern : 'toy.reshape -> (toy.constant)' {
Trying to match "(anonymous namespace)::FoldConstantReshapeOptPattern"
    ** Insert  : 'toy.constant'(0x56545bac1fc0)
    ** Replace : 'toy.reshape'(0x56545baafaf0)
    ** Modified: 'toy.reshape'(0x56545bab0640)
    ** Erase   : 'toy.reshape'(0x56545baafaf0)
"(anonymous namespace)::FoldConstantReshapeOptPattern" result 1
  } -> success : pattern applied successfully
// *** IR Dump After Pattern Application ***
toy.func @main() {
  %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
  %1 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
  %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
  %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
  toy.print %3 : tensor<2x1xf64>
  toy.return
}
```

如果要打印信息，使用op.emitWarning()代替cout以及llvm::errs()，便于调试，有color提示，还可以通过-mlir-print-stacktrace-on-diagnostic打印函数调用栈：

> https://mlir.llvm.org/getting_started/Debugging/

```cpp
op.emitWarning() << "index : "<< index++ <<" call matchAndRewrite transpose op "<< op.getOperationName().str() << "\n";
```

会打印如下信息：
```
/xxx/llvm-project/mlir/test/Examples/Toy/Ch3/transpose_transpose.toy:5:20: warning: index : 1 call matchAndRewrite transpose op toy.transpose
```
## 参考
https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/