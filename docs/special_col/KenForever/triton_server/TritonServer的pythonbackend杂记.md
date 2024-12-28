
## 1 多个版本python

不同的模型包需要使用不同的python版本，比如3.8，3.10.

那么你需要分别为3.8，3.10版本编译python backend stub。

编译参考：[building-custom-python-backend-stub](https://github.com/triton-inference-server/python_backend#building-custom-python-backend-stub

```bash
ls /opt/tritonserver/backends/python/
libtriton_python.so  __pycache__  triton_python_backend_stub  triton_python_backend_utils.py
```

```bash
ldd triton_python_backend_stub
...
libpython3.6m.so.1.0 => /home/ubuntu/envs/miniconda3/envs/python-3-6/lib/libpython3.6m.so.1.0 (0x00007fbb69cf3000)
...
```

libtriton_python.so 是没有链接libpythonxx.so的。
```bash
 ldd /opt/tritonserver/backends/python/libtriton_python.so 
        linux-vdso.so.1 (0x00007ffc6e8df000)
        librt.so.1 => /usr/lib/x86_64-linux-gnu/librt.so.1 (0x00007f9d74414000)
        libtritonserver.so => not found
        libarchive.so.13 => /usr/lib/x86_64-linux-gnu/libarchive.so.13 (0x00007f9d74347000)
        libpthread.so.0 => /usr/lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f9d74324000)
        libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f9d74142000)
        libgcc_s.so.1 => /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f9d74125000)
        libc.so.6 => /usr/lib/x86_64-linux-gnu/libc.so.6 (0x00007f9d73f33000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f9d744d8000)
        libnettle.so.7 => /usr/lib/x86_64-linux-gnu/libnettle.so.7 (0x00007f9d73ef9000)
        libacl.so.1 => /usr/lib/x86_64-linux-gnu/libacl.so.1 (0x00007f9d73eee000)
        liblzma.so.5 => /usr/lib/x86_64-linux-gnu/liblzma.so.5 (0x00007f9d73ec5000)
        libzstd.so.1 => /usr/lib/x86_64-linux-gnu/libzstd.so.1 (0x00007f9d73e1c000)
        liblz4.so.1 => /usr/lib/x86_64-linux-gnu/liblz4.so.1 (0x00007f9d73df9000)
        libbz2.so.1.0 => /usr/lib/x86_64-linux-gnu/libbz2.so.1.0 (0x00007f9d73de6000)
        libz.so.1 => /usr/lib/x86_64-linux-gnu/libz.so.1 (0x00007f9d73dca000)
        libxml2.so.2 => /usr/lib/x86_64-linux-gnu/libxml2.so.2 (0x00007f9d73c10000)
        libm.so.6 => /usr/lib/x86_64-linux-gnu/libm.so.6 (0x00007f9d73ac1000)
        libdl.so.2 => /usr/lib/x86_64-linux-gnu/libdl.so.2 (0x00007f9d73ab9000)
        libicuuc.so.66 => /usr/lib/x86_64-linux-gnu/libicuuc.so.66 (0x00007f9d738d3000)
        libicudata.so.66 => /usr/lib/x86_64-linux-gnu/libicudata.so.66 (0x00007f9d71e12000)
```

并且以不同的命名。比如/opt/tritonserver/backends/python_3_8。
在config.pbtxt中指定backend:"python_3_8", 而不是"python"。

> I don't think Python backend will pick that environment if the Python version is different. The reason is that there is a Python interpreter embedded in the `triton_python_backend_stub` file that is not dependent on the Python version available in the environment (it is always 3.8 unless you are compiling your own stub). If the Python version is different you need to compile your own stub.

> you could create different backend directories and control it using the backend parameter field but you need to put the full Python backend compiled with different Python versions in the backends directory (it must include the stub file, `triton_python_backend_utils` and everything else included). For the environment you need to create a tar file using conda-pack and use that for each of the models.

https://github.com/triton-inference-server/server/issues/3847

通过conda-pack可以在conda中安装不同版本的环境，然后conda-pack导出。注意要设置参数：
```bash
`export PYTHONNOUSERSITE=True`
```
 make sure it is not using dependencies that are not available in the env.

```bash
parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "/home/iman/miniconda3/envs/python-3-6/python3.6.tar.gz"}
}
```



## 2 Malloc set (TcMalloc or jemalloc)

[model-control-mode-explicit](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md#model-control-mode-explicit)

switching from malloc to [tcmalloc](https://github.com/google/tcmalloc) or [jemalloc](https://github.com/jemalloc/jemalloc) by setting the `LD_PRELOAD` environment variable when running Triton, as shown below:

```bash
# Using tcmalloc
LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libtcmalloc.so.4:${LD_PRELOAD} tritonserver --model-repository=/models ...

# Using jemalloc
LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so:${LD_PRELOAD} tritonserver --model-repository=/models ...
```

```bash
# Install tcmalloc
apt-get install gperf libgoogle-perftools-dev

# Install jemalloc
apt-get install libjemalloc-dev
```

## 3 通过EXECUTION_ENV_PATH设置python源码实现

```bash
path_to_activate_ = python_execution_env + "/bin/activate";

path_to_libpython_ = python_execution_env + "/lib";
```

https://github1s.com/triton-inference-server/python_backend/blob/main/src/stub_launcher.cc#L362-L376

如果model_repository_path_路径下存在triton_python_backend_stub，优先使用。设置的EXECUTION_ENV_PATH下的python需要有activate可执行文件。一般是通过conda_pack打包的。

主要包括source激活python环境，设置LD_LIBRARY_PATH，执行triton_python_backend_stub。
```c++
TRITONSERVER_Error*

StubLauncher::Launch(){

// Default Python backend stub

std::string python_backend_stub = python_lib_ + "/triton_python_backend_stub";

  
// Path to alternative Python backend stub

std::string model_python_backend_stub =

std::string(model_repository_path_) + "/triton_python_backend_stub";

  

if (FileExists(model_python_backend_stub)) {

python_backend_stub = model_python_backend_stub;

}


if (python_execution_env_ != "") {
	
	std::stringstream ss;
	
	// Need to properly set the LD_LIBRARY_PATH so that Python environments
	
	// using different python versions load properly.
	
	ss << "source " << path_to_activate_
	
	<< " && exec env LD_LIBRARY_PATH=" << path_to_libpython_
	
	<< ":$LD_LIBRARY_PATH " << python_backend_stub << " " << model_path_
	
	<< " " << shm_region_name_ << " " << shm_default_byte_size_ << " "
	
	<< shm_growth_byte_size_ << " " << parent_pid_ << " " << python_lib_
	
	<< " " << ipc_control_handle_ << " " << stub_name << " "
	
	<< runtime_modeldir_;
	
	ipc_control_->uses_env = true;
	
	bash_argument = ss.str();

}
}
```

### 3.1 插入一个小trick

https://stackoverflow.com/questions/62347343/collect-return-exit-status-of-process-using-system-call-in-linux

system 返回的是wait调用一样的返回值格式，如果要获取status_code需要调用WEXITSTATUS宏定义。
这里的应用，如果返回值不等于1，就表示没有执行权限，采用chmod添加权限。

```cpp
int stub_status_code =

system((python_backend_stub + "> /dev/null 2>&1").c_str());


// If running stub process without any arguments returns any status code,

// other than 1, it can indicate a permission issue as a result of

// downloading the stub process from a cloud object storage service.

if (WEXITSTATUS(stub_status_code) != 1) {

// Give the execute permission for the triton_python_backend_stub to the

// owner.

int error = chmod(python_backend_stub.c_str(), S_IXUSR);

if (error != 0) {

return TRITONSERVER_ErrorNew(

TRITONSERVER_ERROR_INTERNAL,

(std::string("Failed to give execute permission to "

"triton_python_backend_stub in ") +

python_backend_stub + " " + stub_name +

" Error No.: " + std::to_string(error))

.c_str());

}

}
```

https://github.com/triton-inference-server/server/issues/3608


## 4 IPC通信
python backend子进程和triton server通过shm进行ipc进程间通信。采用boost中
```c++
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include "shm_manager.h"
namespace bi = boost::interprocess;
```