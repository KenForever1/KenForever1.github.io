---
title: Tritonserver中shm使用源代码分析
date: 2025-06-21
authors: [KenForever1]
categories: 
  - LLM推理
labels: []
comments: true
---

<!-- more -->

## Tritonserver中shm使用源代码分析

从源码探索Tritonserver中shm如何使用，包括system shm和cuda ipc shm。一个shm req如何传递给server，然后在backend中使用。
在server项目中，初始化了shmmanager。
```c++
// tritonserver/server/src/main.cc
// Manager for shared memory blocks.
auto shm_manager = std::make_shared<triton::server::SharedMemoryManager>();
```
然后初始化GRPC server，将shm_manager传入。
```c++
// Start the HTTP, GRPC, and metrics endpoints.
if (!StartEndpoints(server, trace_manager, shm_manager)) {
exit(1);
}
```
在grpc server中，采用异步方式处理不同类型的请求。分为三种：
```c++
// server/src/grpc/grpc_server.cc
// common请求队列
std::unique_ptr<::grpc::ServerCompletionQueue> common_cq_;
// 模型推理请求队列
std::unique_ptr<::grpc::ServerCompletionQueue> model_infer_cq_;
// 模型流式推理请求队列
std::unique_ptr<::grpc::ServerCompletionQueue> model_stream_infer_cq_;
```
### common 请求处理
在common请求中，每个类型就是一个CommonCallData类型，该类型包括了请求的回调函数。在thread中调用回调函数，处理请求。
commonhandler中注册了如下common请求类型：
```c++
  void RegisterServerLive();
  void RegisterServerReady();
  void RegisterHealthCheck();
  void RegisterModelReady();
  void RegisterServerMetadata();
  void RegisterModelMetadata();
  void RegisterModelConfig();
  void RegisterModelStatistics();
  void RegisterTrace();
  void RegisterLogging();
  void RegisterSystemSharedMemoryStatus();
  void RegisterSystemSharedMemoryRegister();
  void RegisterSystemSharedMemoryUnregister();
  void RegisterCudaSharedMemoryStatus();
  void RegisterCudaSharedMemoryRegister();
  void RegisterCudaSharedMemoryUnregister();
  void RegisterRepositoryIndex();
  void RegisterRepositoryModelLoad();
  void RegisterRepositoryModelUnload();
```
gpu ipc shm和sys shm就在其中的函数注册了, 通过shm_manager注册管理name, raw_handle, byte_size, device_id等。
```c++
void
CommonHandler::RegisterCudaSharedMemoryRegister()
{
  auto OnRegisterCudaSharedMemoryRegister =
      [this](
          ::grpc::ServerContext* ctx,
          inference::CudaSharedMemoryRegisterRequest* request,
          ::grpc::ServerAsyncResponseWriter<
              inference::CudaSharedMemoryRegisterResponse>* responder,
          void* tag) {
        this->service_->RequestCudaSharedMemoryRegister(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteCudaSharedMemoryRegister =
      [this](
          inference::CudaSharedMemoryRegisterRequest& request,
          inference::CudaSharedMemoryRegisterResponse* response,
          ::grpc::Status* status) {
        TRITONSERVER_Error* err = nullptr;
#ifdef TRITON_ENABLE_GPU
        err = shm_manager_->RegisterCUDASharedMemory(
            request.name(),
            reinterpret_cast<const cudaIpcMemHandle_t*>(
                request.raw_handle().c_str()),
            request.byte_size(), request.device_id());
#else
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "failed to register CUDA shared memory region: '" +
                request.name() + "', GPUs not supported")
                .c_str());
#endif  // TRITON_ENABLE_GPU

        GrpcStatusUtil::Create(status, err);
        TRITONSERVER_ErrorDelete(err);
      };

  const std::pair<std::string, std::string>& restricted_kv =
      restricted_keys_.Get(RestrictedCategory::SHARED_MEMORY);
  new CommonCallData<
      ::grpc::ServerAsyncResponseWriter<
          inference::CudaSharedMemoryRegisterResponse>,
      inference::CudaSharedMemoryRegisterRequest,
      inference::CudaSharedMemoryRegisterResponse>(
      "CudaSharedMemoryRegister", 0, OnRegisterCudaSharedMemoryRegister,
      OnExecuteCudaSharedMemoryRegister, false /* async */, cq_, restricted_kv,
      response_delay_);
}
```

## infer grpc请求

### request 处理

```c++
ModelInferHandler::Execute(InferHandler::State* state){
  // Maintain shared pointers(read-only reference) to the shared memory block's
  // information for the shared memory regions used by the request. These
  // pointers will automatically increase the usage count, preventing
  // unregistration of the shared memory. This vector must be cleared in the
  // `InferResponseComplete` callback (after inference) to decrease the count
  // and permit unregistration. The vector will be included in
  // `response_release_payload` for the callback.
  std::vector<std::shared_ptr<const SharedMemoryManager::SharedMemoryInfo>>
      shm_regions_info;

  if (err == nullptr) {
    err = InferGRPCToInput(
        tritonserver_, shm_manager_, request, &serialized_data, irequest,
        &shm_regions_info);
  }
  if (err == nullptr) {
    err = InferAllocatorPayload<inference::ModelInferResponse>(
        tritonserver_, shm_manager_, request, std::move(serialized_data),
        response_queue, &state->alloc_payload_, &shm_regions_info);
  }
}
```

在InferGRPCToInput如果req是shm信息的，就会获取shm信息。然后将shm信息绑定到req的BufferAttributes中。然后调用infer，跳转的schedule模块（抽象类），再交给具体的dynamic scheduler等调度。将request包装成payload，根据调度策略给到model_instance。然后就可以调度到实现的具体infer backend逻辑了。

```c++
void
TritonModelInstance::Execute(
    std::vector<TRITONBACKEND_Request*>& triton_requests)
{
  TRITONBACKEND_ModelInstance* triton_model_instance =
      reinterpret_cast<TRITONBACKEND_ModelInstance*>(this);
  TritonBackend::TritonModelInstanceExecFn_t inst_exec_fn =
      model_->Backend()->ModelInstanceExecFn();

  // If there is an error then we retain ownership of 'requests'
  // and must send error responses.
  TRITONSERVER_Error* err = inst_exec_fn(
      triton_model_instance, &triton_requests[0], triton_requests.size());
  if (err != nullptr) {
    Status status = Status(
        TritonCodeToStatusCode(TRITONSERVER_ErrorCode(err)),
        TRITONSERVER_ErrorMessage(err));
    for (TRITONBACKEND_Request* tr : triton_requests) {
      std::unique_ptr<InferenceRequest> ur(
          reinterpret_cast<InferenceRequest*>(tr));
      InferenceRequest::RespondIfError(ur, status, true /* release_requests */);
    }

    TRITONSERVER_ErrorDelete(err);
  }
}
```

### response 处理
ModelInferHandler类的构造函数中，注册了回调函数，将ipc信息传递给了Response的allocator。就可以通过ipc返回response给client。
```c++
    // Create the allocator that will be used to allocate buffers for
    // the result tensors.
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorNew(
            &allocator_, InferResponseAlloc, InferResponseFree,
            InferResponseStart),
        "creating inference response allocator");
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorSetQueryFunction(
            allocator_, OutputBufferQuery),
        "setting allocator's query function");
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorSetBufferAttributesFunction(
            allocator_, OutputBufferAttributes),
        "setting allocator's output buffer attributes function");
```
## 使用shm extention

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_shared_memory.html

要使用shm extention，让req和response传输shm的fd，而不是直接传输数据。

需要client以http或者grpc的方式发送请求，注册shm区域信息，后面的请求和response就可以直接传输shm的fd使用了。

在我们编写的backend中可以通过backend api获取request的shm信息。
```c++
TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_RequestOutputBufferProperties(
    TRITONBACKEND_Request* request, const char* name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  InferenceRequest* tr = reinterpret_cast<InferenceRequest*>(request);
  auto status =
      tr->OutputBufferProperties(name, byte_size, memory_type, memory_type_id);
  if (!status.IsOk()) {
    return TRITONSERVER_ErrorNew(
        StatusCodeToTritonCode(status.StatusCode()), status.Message().c_str());
  }
  return nullptr;  // success
}



TRITONBACKEND_DECLSPEC TRITONSERVER_Error* TRITONBACKEND_InputBufferAttributes(
    TRITONBACKEND_Input* input, const uint32_t index, const void** buffer,
    TRITONSERVER_BufferAttributes** buffer_attributes);


/// Get the memory type field of the buffer attributes.
///
/// \param buffer_attributes The buffer attributes object.
/// \param memory_type Returns the memory type associated with the buffer
/// attributes object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_BufferAttributesMemoryType(
    TRITONSERVER_BufferAttributes* buffer_attributes,
    TRITONSERVER_MemoryType* memory_type);

/// Get the CudaIpcHandle field of the buffer attributes object.
///
/// \param buffer_attributes The buffer attributes object.
/// \param cuda_ipc_handle Returns the memory type associated with the buffer
/// attributes object. If the cudaIpcHandle does not exist for the buffer,
/// nullptr will be returned.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_BufferAttributesCudaIpcHandle(
    TRITONSERVER_BufferAttributes* buffer_attributes, void** cuda_ipc_handle);

/// Get the byte size field of the buffer attributes.
///
/// \param buffer_attributes The buffer attributes object.
/// \param byte_size Returns the byte size associated with the buffer attributes
/// object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_DECLSPEC TRITONSERVER_Error* TRITONSERVER_BufferAttributesByteSize(
    TRITONSERVER_BufferAttributes* buffer_attributes, size_t* byte_size);

```
然后通过shm获得数据，进行处理，再生成response。

举例：
比如在python backend的实现中：
```c++
TRITONSERVER_Error*
ModelInstanceState::GetInputTensor(
    const uint32_t input_idx, std::shared_ptr<PbTensor>& input_tensor,
    TRITONBACKEND_Request* request,
    std::shared_ptr<std::vector<TRITONBACKEND_Response*>>& responses)
{
    TRITONSERVER_BufferAttributes* buffer_attributes;

      // This value is not used.
      const void* buffer_p;
      RETURN_IF_ERROR(TRITONBACKEND_InputBufferAttributes(
          in, 0, &buffer_p, &buffer_attributes));

      input_tensor = std::make_shared<PbTensor>(
          std::string(input_name),
          std::vector<int64_t>(input_shape, input_shape + input_dims_count),
          input_dtype, src_memory_type, src_memory_type_id,
          const_cast<void*>(buffer), input_byte_size,
          nullptr /* DLManagedTensor */);

      cudaIpcMemHandle_t* cuda_ipc_handle;
      RETURN_IF_ERROR(TRITONSERVER_BufferAttributesCudaIpcHandle(
          buffer_attributes, reinterpret_cast<void**>(&cuda_ipc_handle)));
      if (cuda_ipc_handle != nullptr) {
        RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(
            Stub()->ShmPool(), false /* copy_gpu */));
        RETURN_IF_EXCEPTION(
            input_tensor->Memory()->SetCudaIpcHandle(cuda_ipc_handle));
      } else {
        RETURN_IF_EXCEPTION(input_tensor->SaveToSharedMemory(
            Stub()->ShmPool(), true /* copy_gpu */));
      }
}
```