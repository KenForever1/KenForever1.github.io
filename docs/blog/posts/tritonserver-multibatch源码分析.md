---
title: tritonserver-multibatch源码分析
date: 2025-06-21
authors: [KenForever1]
categories: 
  - LLM推理
labels: []
comments: true
---

<!-- more -->
## 问题

```c++
name: "python_batched_service"

backend: "python"
max_batch_size: 16

dynamic_batching {
  max_queue_delay_microseconds: 500000
}

input [
  {
    name: "input_field"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
```
对于一个输入为input_field的模型，如果输入的shape不同，比如[1,20]和[1,30]、[1、40]，那么dynamic batching是否会凑batch？

答案是不会，只有连续发送的[1,20]这种相同输入shape的请求才会凑batch。不一样的叫不规则输入，也叫ragged batch。

可以参考python例子。

https://github.com/triton-inference-server/server/issues/6937

https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_statistics.md
### tritonserver

不同shape，采用dynamic batching方式不会凑batch。

需要采用ragged batching方式。或者将proto3的int32改成fixed32，以及避免传输0。

```c++
// repo-core-src/src/backend_model.cc
Status
TritonModel::SetConfiguredScheduler()
{
  std::unique_ptr<Scheduler> scheduler;

  // Need to enforce equal shape batches (i.e. non-ragged batches) if
  // the model 1) allows one or more variable-size input tensors that
  // are not marked as 'allow_ragged_batch' or 2) has one or more
  // shape-tensor inputs. This is not needed if all input shapes are
  // non-variable and if there are no shape tensors... so we don't
  // enable it in that case for efficiency reasons.
  std::unordered_map<std::string, bool> enforce_equal_shape_tensors;
  for (const auto input : config_.input()) {
    // https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#shape-tensors, 目前只有TensorRT支持shape_tensor。
    if (input.is_shape_tensor()) {
      enforce_equal_shape_tensors.insert({input.name(), true});
    } else if (
        !input.allow_ragged_batch() &&
        (triton::common::GetElementCount(input) == -1)) {
      enforce_equal_shape_tensors.insert({input.name(), false});
    }
  }
    // ......

    RETURN_IF_ERROR(DynamicBatchScheduler::Create(
    this, nullptr, 0 /*nice*/, true /* dynamic_batching_enabled */,
    config_.max_batch_size(), enforce_equal_shape_tensors,
    config_.dynamic_batching(),
    config_.response_cache().enable() /* response_cache_enable */,
    &scheduler));
}

```

```c++
// _deps/repo-core-src/src/dynamic_batch_scheduler.cc
uint64_t
DynamicBatchScheduler::GetDynamicBatch(){

    // When there is optional input or input shape must be enforced,
    // the inputs in the requests must be examined for forming a batch
    const bool check_input =
        !enforce_equal_shape_tensors_.empty() || has_optional_input_;
    
    // If there is no pending batch, then this request is starting a
    // new batch.
    if ((payload_batch_size + queue_.PendingBatchCount()) == 0) {
      // Get the shape of the new batch that is being started...
      if (check_input) {
        if (!curr_payload_->MutableRequiredEqualInputs()
                 ->Initialize(
                     queue_.RequestAtCursor(), enforce_equal_shape_tensors_,
                     has_optional_input_)
                 .IsOk()) {
          send_now = true;
          break;
        }
      }
    } else {
      // There is a pending batch and adding this request would make
      // the batch size larger than all of the preferred batch sizes,
      // so mark the cursor at this point. Not sending the pending batch so
      // that we can examine the queue delay of requests that fits in a batch.
      if (((payload_batch_size + pending_batch_size_ + batch_size) >
           max_preferred_batch_size_) &&
          (best_preferred_batch_size == 0)) {
        best_preferred_batch_size = pending_batch_size_;
        queue_.MarkCursor();
        payload_saturated_ = true;
      }
      if ((payload_batch_size + pending_batch_size_ + batch_size) >
          max_batch_size_) {
        send_now = true;
        break;
      }

      // There is a pending batch and it has a different shape then
      // this request, so send the pending batch as it is.
      if (check_input &&
          !curr_payload_->MutableRequiredEqualInputs()->HasEqualInputs(
              queue_.RequestAtCursor())) {
        curr_payload_->MarkSaturated();
        send_now = true;
        break;
      }
    }

    // 直接发送
    // If the delay has been exceeded, or if the current batch can't grow
    // any larger then just immediately execute whatever is pending.
    if (send_now || ((payload_batch_size + pending_batch_size_) >=
                    max_preferred_batch_size_)) {
        payload_saturated_ = true;
        return 0;
    }
}
```

```c++
bool
RequiredEqualInputs::HasEqualInputs(
    const std::unique_ptr<InferenceRequest>& request){
        //......
        const auto& d1 = itr->second.first->Data();
          const auto& d2 = input->Data();

          // For now being conservative and assuming that content
          // comparison is for shape tensors which are likely to always
          // be in a single buffer.
          if ((d1->BufferCount() != 1) || (d2->BufferCount() != 1)) {
            return false;
          }

          size_t d1_byte_size, d2_byte_size;
          TRITONSERVER_MemoryType d1_memory_type, d2_memory_type;
          int64_t d1_memory_id, d2_memory_id;
          const char* d1_buffer = d1->BufferAt(
              0 /* idx */, &d1_byte_size, &d1_memory_type, &d1_memory_id);
          const char* d2_buffer = d2->BufferAt(
              0 /* idx */, &d2_byte_size, &d2_memory_type, &d2_memory_id);

          // Tensor must be same size and in in CPU memory so that it
          // can be easily compared. If not return false conservatively.
          if ((d1_byte_size != d2_byte_size) || (d1_buffer == nullptr) ||
              (d2_buffer == nullptr) ||
              (d1_memory_type == TRITONSERVER_MEMORY_GPU) ||
              (d2_memory_type == TRITONSERVER_MEMORY_GPU)) {
            return false;
          }

          if (strncmp(d1_buffer, d2_buffer, d1_byte_size) != 0) {
            return false;
          }
    }
```

### 采用ragged batch 

```
max_batch_size: 16
input [
  {
    name: "INPUT"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
```
改为：
```
max_batch_size: 16
input [
  {
    name: "INPUT"
    data_type: TYPE_FP32
    dims: [ -1 ]
    allow_ragged_batch: true
  }
]
```

python backend和一般自定义的backend都不需要加：
```
batch_input [
  {
    kind: BATCH_ACCUMULATED_ELEMENT_COUNT
    target_name: "INDEX"
    data_type: TYPE_FP32
    source_input: "INPUT"
  }
]
```
> The backends, such as ONNX Runtime backend, TensorFlow backend, PyTorch backend, and TensorRT backend, require models to accept ragged inputs as 1-dimensional tensors. These backends concatenates the request inputs into the 1-dimensional tensor.Because the concatenated input doesn’t track the start and end index for each request, the backends often require the model to have additional input(s), batch input, that describe various information about the batch formed.
> https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ragged_batching.html

像ONNX Runtime backend, TensorFlow backend, PyTorch backend, and TensorRT backend，将request inputs 拼接成1维tensor，但是这个拼接的tensor无法跟踪每个request的start和end index，所以需要额外的batch input来描述batch的构成。

可以查看实现：
```c++
// https://github1s.com/triton-inference-server/onnxruntime_backend/blob/main/src/onnxruntime.cc
```

### proto3

proto的规则，int32类型的0值序列化不占用字节数，非0编码采用variant，数字根据大小不同，在2-5个字节变化。

```c++
syntax = "proto3";

package hello.proto;
message Image {
    int32 device_id = 1;
    int32 fd = 2;
    bytes data = 3;
}
message ImageSet {
    repeated Image images = 1;
}
```
需要改成：
```c++
syntax = "proto3";

package hello.proto;
message Image {
    fixed32 device_id = 1;
    fixed32 fd = 2;
    bytes data = 3;
}
message ImageSet {
    repeated Image images = 1;
}
```
fixed32只能解决非0值的序列化后长度保持一致，同时需要避免device_id和fd传递0，比如可以先加1，从1开始。传递到server后再减1。

## python backend例子
model.py
```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        num_requests = len(requests)
        print(f"Received a batch containing {num_requests} request(s)")

        responses = [
            pb_utils.InferenceResponse(output_tensors=[])
            for _ in range(num_requests)
        ]
        return responses

    def finalize(self):
        print("Cleaning up...")
```

client.py
```python
import tritonclient.grpc as grpcclient
import numpy as np


triton_client = grpcclient.InferenceServerClient(
    url="localhost:8001",
    verbose=False,
)


def callback(result, error):
    if error:
        print(error)
    pass


def send_single_request(array):
    inputs = [
        grpcclient.InferInput("input_field", array.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(array)

    triton_client.async_infer(
        model_name="python_batched_service",
        inputs=inputs,
        model_version="1",
        outputs=[],
        callback=callback,
    )

def send_many_requests(input_sizes):
    for ix in range(200):
        size = input_sizes[ix % len(input_sizes)]
        array = np.random.rand(*size).astype(np.float32)
        send_single_request(array=array)


if __name__ == "__main__":
    send_many_requests(input_sizes=[
        (1, 20,),
    ])

    send_many_requests(input_sizes=[
        (1, 20,),
        (1, 40),
        (1, 60),
    ])
```