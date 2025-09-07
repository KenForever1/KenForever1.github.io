---
title: tritonserver_python_backend_shm_com源码分析
date: 2025-06-21
authors: [KenForever1]
categories: 
  - LLM推理
labels: []
comments: true
---
<!-- more -->

### 背景介绍

tritonserver作为一个AI推理服务，支持python backend的方式，通过python 子进程方式运行模型或者自定义的逻辑。

简单来说，就是通过fork启动了一个stub子进程，stub子进程通过pybind11运行用户子定义的python代码。当c++实现的tritonserver收到请求后，就会调用python脚本，经过处理后返回结果。

+ fork stub子进程，执行用户自定义的python代码
+ tritonserver作为父进程，需要和stub子进程通信，比如控制stub停止、发送请求、控制命令等。

这就设计了进程之间的通信，如果通过网络通信，性能会下降，这里采用了shm（共享内存）的方式。

本文介绍了tritonserver中，如何通过共享内存和子进程通信，主要介绍共享内存管理的封装以及c++ boost inprocess模块的使用。这块功能是可以复用的，可以用于其他场景。

### 共享内存和消息队列的设计

这里进程间通信，设计了一个消息队列，承载消息队列的载体是共享内存，即通过共享内存来存储消息队列。

既然是跨进程间传输数据的消息队列，那么肯定两个进程都会获取同一个消息队列的句柄，生产者创建（create）并且发送（push）消息，消费者load消息队列并且消费（pop）消息。

### 从parent进程视角看

parent进程创建了消息队列，push消息到消息队列stub_message_queue_，然后等待子进程消费消息。然后从parent_message_queue_接受子进程返回的消息。

#### 共享内存以及消息队列的创建

例如下面是MessageQueue类创建的几种不同的消息队列，用于parent进程和stub进程之间的通信，消息队列的类型是bi::managed_external_buffer::handle_t。

```c++
// 源码地址：https://github.com/triton-inference-server/python_backend/blob/main/src/stub_launcher.cc#L151
// stub_message_queue_是parent发送控制信息给stub进程的队列，比如发送初始化、停止等commond
RETURN_IF_EXCEPTION(
      stub_message_queue_ =
          MessageQueue<bi::managed_external_buffer::handle_t>::Create(
              shm_pool_, shm_message_queue_size_));
// parent_message_queue_是stub进程发送信息给parent进程的队列
  RETURN_IF_EXCEPTION(
      parent_message_queue_ =
          MessageQueue<bi::managed_external_buffer::handle_t>::Create(
              shm_pool_, shm_message_queue_size_));
  RETURN_IF_EXCEPTION(
      stub_to_parent_mq_ =
          MessageQueue<bi::managed_external_buffer::handle_t>::Create(
              shm_pool_, shm_message_queue_size_));
  RETURN_IF_EXCEPTION(
      parent_to_stub_mq_ =
          MessageQueue<bi::managed_external_buffer::handle_t>::Create(
              shm_pool_, shm_message_queue_size_));
```

SharedMemoryManager类是一个可以复用的类，封装了共享内存的创建和销毁，以及共享内存中对象的构造和析构。这里指定了共享内存的名称，大小，增长大小，是否创建等参数，为进程通信创建了一段shm共享内存。

并且在共享内存中创建了IPCControlShm对象，这个对象包含了消息队列的句柄，以及一些进程健康状态的标志位。

```c++
std::unique_ptr<SharedMemoryManager> shm_pool_ = std::make_unique<SharedMemoryManager>(
        shm_region_name_, shm_default_byte_size_, shm_growth_byte_size_,
        true /* create */);
AllocatedSharedMemory<IPCControlShm> current_ipc_control =
      shm_pool_->Construct<IPCControlShm>();
```

IPCControlShm结构体包含了进程通信需要的所有信息，包括两个进程的健康状态，两个进程的消息队列句柄等。
```c++
// Control data structure for the communication between the Python stub and the
// main stub.
struct IPCControlShm {
  bool stub_health;
  bool parent_health;
  bool uses_env;
  bool decoupled;
  bi::interprocess_mutex parent_health_mutex;
  bi::interprocess_mutex stub_health_mutex;
  bi::managed_external_buffer::handle_t stub_message_queue;
  bi::managed_external_buffer::handle_t parent_message_queue;
  bi::managed_external_buffer::handle_t stub_to_parent_mq;
  bi::managed_external_buffer::handle_t parent_to_stub_mq;
  bi::managed_external_buffer::handle_t memory_manager_message_queue;
};
```

##### 发送消息

```c++
std::unordered_map<std::string, std::string> initialize_map = {
    {"model_config", model_config_buffer_.MutableContents()},
    {"model_instance_kind", kind_},
    {"model_instance_name", model_instance_name_},
    {"model_instance_device_id", std::to_string(device_id_)},
    {"model_repository", model_repository_path_},
    {"model_version", std::to_string(model_version_)},
    {"model_name", model_name_}};

std::unique_ptr<IPCMessage> initialize_message =
    IPCMessage::Create(shm_pool_, false /* inline_response */);
initialize_message->Command() = PYTHONSTUB_InitializeRequest;

std::unique_ptr<PbMap> pb_map = PbMap::Create(shm_pool_, initialize_map);
bi::managed_external_buffer::handle_t initialize_map_handle =
    pb_map->ShmHandle();

initialize_message->Args() = initialize_map_handle;
stub_message_queue_->Push(initialize_message->ShmHandle());
```

##### 接收消息

```c++
bi::managed_external_buffer::handle_t message;
RETURN_IF_ERROR(ReceiveMessageFromStub(message, initialization_timeout_ms));

std::unique_ptr<IPCMessage> initialize_response_message =
    IPCMessage::LoadFromSharedMemory(shm_pool_, message);

if (initialize_response_message->Command() != PYTHONSTUB_InitializeResponse) {
return TRITONSERVER_ErrorNew(
    TRITONSERVER_ERROR_INTERNAL,
    (std::string(
            "Received unexpected response from Python backend stub: ") +
        model_instance_name_)
        .c_str());
}

auto initialize_response =
    std::move((shm_pool_->Load<InitializeResponseShm>(
                initialize_response_message->Args())))
        .data_;
```

ReceiveMessageFromStub实际上就是从消息队列parent_message_queue_中取出消息。
```c++
message = parent_message_queue_->Pop(
    timeout_miliseconds /* duration ms */, success);
```

### 从stub进程视角看

从stub进程，也就是python子进程视角看，它就不是创建共享内存了。而是使用parent进程创建的共享内存，然后通过LoadFromSharedMemory方法load创建好的消息队列，直接使用。
```c++
// https://github.com/triton-inference-server/python_backend/blob/main/src/pb_stub.cc#L164
shm_pool_ = std::make_unique<SharedMemoryManager>(
    shm_region_name, shm_default_size, shm_growth_size, false /* create */);

AllocatedSharedMemory<IPCControlShm> ipc_control =
    shm_pool_->Load<IPCControlShm>(ipc_control_handle);
ipc_control_ = ipc_control.data_.get();

stub_message_queue_ = MessageQueue<bi::managed_external_buffer::handle_t>::
    LoadFromSharedMemory(shm_pool_, ipc_control_->stub_message_queue);

```
上面说了parent给stub发送了一个initialize_message，stub进程收到后，会回复一个initialize_response。子进程通过Pop从stub_message_queue_接受消息。

#### 接收消息
```c++
std::unique_ptr<IPCMessage>
Stub::PopMessage()
{
  bool success = false;
  std::unique_ptr<IPCMessage> ipc_message;
  bi::managed_external_buffer::handle_t message;
  while (!success) {
    message = stub_message_queue_->Pop(1000, success);
  }

  ipc_message = IPCMessage::LoadFromSharedMemory(shm_pool_, message);

  return ipc_message;
}
```
stub进程的核心逻辑就是调用RunCommand方法，这个方法会从stub_message_queue_中取出消息，然后根据消息类型执行不同的逻辑。

```c++
std::unique_ptr<IPCMessage> ipc_message;
{
// Release the GIL lock when waiting for new message. Without this line, the
// other threads in the user's Python model cannot make progress if they
// give up GIL.
py::gil_scoped_release release;
ipc_message = this->PopMessage();
}
switch (ipc_message->Command()) {
case PYTHONSTUB_CommandType::PYTHONSTUB_InitializeRequest: 
//...
}
```
#### pybind11调用python逻辑

stub是一个c++实现的进程，它收到initialize_message后，会通过pybind11库。pybind11库可以调用用户实现的python逻辑，比如获取python类、调用类功能函数。这里简单放一下代码：

```c++
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
void
Stub::Initialize(bi::managed_external_buffer::handle_t map_handle)
{
  py::module sys = StubSetup();

  py::module python_backend_utils =
      py::module_::import("triton_python_backend_utils");
  py::module c_python_backend_utils =
      py::module_::import("c_python_backend_utils");
  // ......
  c_python_backend_utils.attr("shared_memory") = py::cast(shm_pool_.get());

  async_event_loop_ = py::none();
  background_futures_ = py::set();

  // 创建一个TritonPythonModel对象，这个对象是用户实现的python类，用户需要在类中实现initialize、execute、finalizer等函数
  py::object TritonPythonModel = sys.attr("TritonPythonModel");
  model_instance_ = TritonPythonModel();

  std::unordered_map<std::string, std::string> map;
  std::unique_ptr<PbMap> pb_map_shm =
      PbMap::LoadFromSharedMemory(shm_pool_, map_handle);
  // Get the unordered_map representation of the map in shared memory.
  map = pb_map_shm->UnorderedMap();
  py::dict model_config_params;
  for (const auto& pair : map) {
    model_config_params[pair.first.c_str()] = pair.second;
  }

  // Call initialize if exists.
  // 调用用户实现的initialize函数
  if (py::hasattr(model_instance_, "initialize")) {
    model_instance_.attr("initialize")(model_config_params);
  }
}
```

### 总结

本文由于篇幅限制，介绍了triton的IPC通信机制，以及stub进程的初始化过程，以及介绍stub进程如何执行用户实现的python逻辑。其中的进程间通信机制和pybind11调用python逻辑，可以复用到你自己的项目中。这种跨语言调用很常见，常见的还有c++中调用lua、调用js等。

消息队列使用到了SharedMemoryManager类，该类采用boost.interprocess库实现。关于[消息队列的实现](https://github.com/triton-inference-server/python_backend/blob/main/src/message_queue.h#L67)以及[共享内存的boost.interprocess库的使用](https://github.com/triton-inference-server/python_backend/blob/main/src/shm_manager.h#L44)，可以参考源代码实现。后续文章可能会展开介绍。
