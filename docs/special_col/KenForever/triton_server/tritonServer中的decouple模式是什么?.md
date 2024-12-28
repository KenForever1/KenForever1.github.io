# Triton推理服务器中的解耦模式

在Triton推理服务器中，解耦模式允许用户针对一个请求发送多个响应，或者不发送任何响应。此外，模型还可以按照与请求批次执行顺序不同的顺序发送响应。这类模型被称为解耦模型。要使用这种模式，模型配置中的事务策略必须设置为解耦模式。

在解耦模式下，模型必须为每个请求使用`InferenceResponseSender`对象，以便为该请求持续创建和发送任意数量的响应。此模式下的工作流程可能如下：

1. `execute`函数接收一个`pb_utils.InferenceRequest`的批次，作为长度为N的数组。

2. 迭代每个`pb_utils.InferenceRequest`对象，并为每个对象执行以下步骤：

   - 使用`InferenceRequest.get_response_sender()`获取该请求的`InferenceResponseSender`对象。
   
   - 创建并填充要发送回的`pb_utils.InferenceResponse`。
   
   - 使用`InferenceResponseSender.send()`发送上述响应。如果这是最后一个请求，则在`InferenceResponseSender.send()`中传递`pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL`作为标志。否则，继续执行步骤1以发送下一个请求。

在这种模式下，`execute`函数的返回值应为`None`。

与上述类似，如果某个请求出现错误，可以使用`TritonError`对象为该特定请求设置错误消息。在为`pb_utils.InferenceResponse`对象设置错误后，使用`InferenceResponseSender.send()`将带有错误的响应发送回用户。

从23.10版本开始，可以直接在`InferenceResponseSender`对象上使用`response_sender.is_cancelled()`检查请求取消状态。即使请求已取消，仍需要在响应末尾发送`TRITONSERVER_RESPONSE_COMPLETE_FINAL`标志。

## 1 使用案例

解耦模式功能强大，支持多种其他使用场景：

- 如果模型不需要为请求发送任何响应，则调用`InferenceResponseSender.send()`时不带响应，但标志参数设置为`pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL`。
  
- 模型可以以不同于接收请求的顺序发送响应。
  
- 请求数据和`InferenceResponseSender`对象可以传递给模型中的单独线程。这意味着主调用线程可以从`execute`函数中退出，而模型可以继续生成响应，只要它持有`InferenceResponseSender`对象即可。

解耦示例展示了可以从解耦API中实现的全部功能。请阅读解耦后端和模型的详细信息，以了解如何托管解耦模型。

https://github.com/triton-inference-server/python_backend?tab=readme-ov-file#decoupled-mode