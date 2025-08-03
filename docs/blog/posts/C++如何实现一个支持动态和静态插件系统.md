---
title: C++如何实现一个支持动态、静态插件的系统
date: 2025-08-03
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

<!-- more -->

## 引言

通过[英伟达推理传输库（NIXL）](https://github.com/ai-dynamo/nixl)学习C++如何实现一个支持动态、静态插件注册和管理！

NIXL是一个开源项目，英伟达推理传输库（NIXL）旨在加速诸如英伟达Dynamo等人工智能推理框架中的点对点通信，同时通过模块化插件架构，为各种类型的内存（如CPU和GPU）和存储（如文件、块存储和对象存储）提供抽象。

+ 支持Python、Rust语言绑定。python采用pybind11实现，Rust采用bindgen实现。
+ 采用meson构建系统进行构建。
+ 采用插件机制，区分了动态插件（dlsym动态库加载）和静态插件（编译时链接）。
+ 接口api实现清晰，接口和实现分离，方便进行跨语言FFI绑定。

## C++插件系统接口和管理

一个插件系统需要如何考虑接口实现呢？需要包括哪些信息？

+ 插件版本定义，在插件加载时可以对插件版本进行校验，确保插件和主程序兼容。

+ 通过 __attribute__((visibility("default"))) 导出接口函数，使得动态库加载时可以找到这些函数。显式声明需要对外暴露的接口，避免默认导出所有符号，确保关键函数或变量可被外部链接。

### 插件接口定义

在一个api接口头文件中定义插件接口，包括插件版本、插件名称、插件初始化、插件销毁等函数。在项目的组织结构中，可以专门放到一个api目录下。
```c++
#include "backend/backend_engine.h"

// Define the plugin API version
#define NIXL_PLUGIN_API_VERSION 1

// Define the plugin interface class
class nixlBackendPlugin {
public:
    int api_version;

    // Function pointer for creating a new backend engine instance
    nixlBackendEngine* (*create_engine)(const nixlBackendInitParams* init_params);

    // Function pointer for destroying a backend engine instance
    void (*destroy_engine)(nixlBackendEngine* engine);

    // Function to get the plugin name
    const char* (*get_plugin_name)();

    // Function to get the plugin version
    const char* (*get_plugin_version)();

    // Function to get backend options
    nixl_b_params_t (*get_backend_options)();

    // Function to get supported backend mem types
    nixl_mem_list_t (*get_backend_mems)();
};

// Macro to define exported C functions for the plugin
#define NIXL_PLUGIN_EXPORT __attribute__((visibility("default")))

// Creator Function type for static plugins
typedef nixlBackendPlugin* (*nixlStaticPluginCreatorFunc)();

// Plugin must implement these functions for dynamic loading
extern "C" {
    // Initialize the plugin
    NIXL_PLUGIN_EXPORT nixlBackendPlugin* nixl_plugin_init();

    // Cleanup the plugin
    NIXL_PLUGIN_EXPORT void nixl_plugin_fini();
}
```

### 插件管理器

通过一个单例，进行插件管理。
```c++
nixlPluginManager& nixlPluginManager::getInstance() {
    // Meyers singleton initialization is safe in multi-threaded environment.
    // Consult standard [stmt.dcl] chapter for details.
    static nixlPluginManager instance;
    return instance;
}
```
看一下nixlPluginManager的成员变量定义：
```c++
using nixl_backend_t = std::string;

class nixlPluginManager {
private:
    std::map<nixl_backend_t, std::shared_ptr<const nixlPluginHandle>> loaded_plugins_; // 同时保存了注册的动态插件和静态插件信息
    std::vector<std::string> plugin_dirs_; // 保存插件搜索路径
    std::vector<nixlStaticPluginInfo> static_plugins_; // 保存有静态插件的信息，方便在unload时进行过滤静态插件，因为静态插件不需要dlclose。
    // ......
}
```

### 维护插件信息

nixlPluginHandle同时支持保存动态插件和静态插件的信息。nixlPluginHandle结构只在构造和析构时修改，在查询和插件实例化时只读。因此可以在多线程中安全使用，不需要加锁。

```c++
class nixlPluginHandle {
private:
    void* handle_;         // Handle to the dynamically loaded library
    nixlBackendPlugin* plugin_;  // Plugin interface

public:
    nixlPluginHandle(void* handle, nixlBackendPlugin* plugin);
    ~nixlPluginHandle();

    nixlBackendEngine* createEngine(const nixlBackendInitParams* init_params) const;
    void destroyEngine(nixlBackendEngine* engine) const;
    const char* getName() const;
    const char* getVersion() const;
    nixl_b_params_t getBackendOptions() const;
    nixl_mem_list_t getBackendMems() const;
};
```

## 静态插件和动态插件注册

### 静态插件

```c++
// Creator Function for static plugins
typedef nixlBackendPlugin* (*nixlStaticPluginCreatorFunc)();


// Structure to hold static plugin info
struct nixlStaticPluginInfo {
    const char* name;
    nixlStaticPluginCreatorFunc createFunc;
};
```

注册静态插件: 静态插件只需要保存一个name和createFunc，并且在注册时，执行createFunc进行实例化（一个静态全局实例）。然后将nixlBackendPlugin*信息保存到loaded_plugins_的map中。
```c++
void nixlPluginManager::registerStaticPlugin(const char* name, nixlStaticPluginCreatorFunc creator) {
    lock_guard lg(lock);

    nixlStaticPluginInfo info;
    info.name = name;
    info.createFunc = creator;
    static_plugins_.push_back(info);

    //Static Plugins are considered pre-loaded
    nixlBackendPlugin* plugin = info.createFunc();
    NIXL_INFO << "Loading static plugin: " << name;
    if (plugin) {
        // Register the loaded plugin
        auto plugin_handle = std::make_shared<const nixlPluginHandle>(nullptr, plugin);
        loaded_plugins_[name] = plugin_handle;
    }
}
```

静态插件不需要unload, 相比动态插件需要dlclose。

在编译时，决定是否注册静态插件。createStaticPosixPlugin函数在插件模块中实现（后面实现环节有介绍），这里通过宏控制是否注册静态插件。
```c++
#ifdef STATIC_PLUGIN_POSIX
        extern nixlBackendPlugin* createStaticPosixPlugin();
        registerStaticPlugin("POSIX", createStaticPosixPlugin);
#endif // STATIC_PLUGIN_POSIX

#ifdef STATIC_PLUGIN_GDS
#ifndef DISABLE_GDS_BACKEND
        extern nixlBackendPlugin* createStaticGdsPlugin();
        registerStaticPlugin("GDS", createStaticGdsPlugin);
#endif // DISABLE_GDS_BACKEND
#endif // STATIC_PLUGIN_GDS
```

### 动态插件加载

通过一个void *handle，保存动态库的句柄。

支持设置插件搜索路径，并加载插件。在加载插件时，先根据插件名称校验是否已经为静态插件，即已经preload，可以直接返回静态插件的handle。

在插件加载时，会进行插件版本校验。

```c++
std::shared_ptr<const nixlPluginHandle> nixlPluginManager::loadPluginFromPath(const std::string& plugin_path) {
    // Open the plugin file
    void* handle = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        NIXL_ERROR << "Failed to load plugin from " << plugin_path << ": " << dlerror();
        return nullptr;
    }

    // Get the initialization function
    typedef nixlBackendPlugin* (*init_func_t)();
    init_func_t init = (init_func_t) dlsym(handle, "nixl_plugin_init");
    if (!init) {
        NIXL_ERROR << "Failed to find nixl_plugin_init in " << plugin_path << ": " << dlerror();
        dlclose(handle);
        return nullptr;
    }

    // Call the initialization function
    nixlBackendPlugin* plugin = init();
    if (!plugin) {
        NIXL_ERROR << "Plugin initialization failed for " << plugin_path;
        dlclose(handle);
        return nullptr;
    }

    // Check API version
    if (plugin->api_version != NIXL_PLUGIN_API_VERSION) {
        NIXL_ERROR << "Plugin API version mismatch for " << plugin_path
                   << ": expected " << NIXL_PLUGIN_API_VERSION
                   << ", got " << plugin->api_version;
        dlclose(handle);
        return nullptr;
    }

    // Create and store the plugin handle
    auto plugin_handle = std::make_shared<const nixlPluginHandle>(handle, plugin);

    return plugin_handle;
}
```

介绍到现在，一个插件的框架已经搭建好了。支持插件管理、插件接口定义、动态和静态插件加载。
接下来，就是根据插件接口，实现具体的插件。如何实现支持动态加载的插件和静态插件。

## 如何实现插件Plugin

实现插件只需要导入头文件，并实现插件接口。保证了插件接口的统一性，结构组织的清晰。
```c++
#include "backend/backend_plugin.h"

```
这个Posix是英伟达推理传输库（NIXL）中其中一个插件实现，其他插件还包括了hf3fs、mooncake、cuda_gds等。看一下如何实现一个Posix插件，下面是对插件接口函数的实现：
```c++
// Function to create a new POSIX backend engine instance
static nixlBackendEngine* create_posix_engine(const nixlBackendInitParams* init_params) {
    return new nixlPosixEngine(init_params);
}

// 资源的正确释放
static void destroy_posix_engine(nixlBackendEngine *engine) {
    delete engine;
}

// Function to get the plugin name，插件名称
static const char* get_plugin_name() {
    return "POSIX";
}

// Function to get the plugin version，插件版本
static const char* get_plugin_version() {
    return "0.1.0";
}

// Function to get backend options
static nixl_b_params_t get_backend_options() {
    nixl_b_params_t params;
    return params;
}

// Function to get supported backend mem types
static nixl_mem_list_t get_backend_mems() {
    return {DRAM_SEG, FILE_SEG};
}

```

静态插件和动态插件实现的区别：

+ 静态插件在编译时，通过宏控制是否注册静态插件，在程序编译时，根据宏定义，决定是否调用createStaticPosixPlugin函数。将静态全局变量nixlBackendPlugin*信息保存到loaded_plugins_的map中。

+ 动态插件在运行时，通过dlopen加载动态库，并调用nixl_plugin_init函数，初始化插件。

```c++
#ifdef STATIC_PLUGIN_POSIX

// 通过静态全局变量，注册静态插件实例，在程序编译时，根据宏定义，决定是否调用createStaticPosixPlugin函数
// 决定是否注册静态插件
// Static plugin structure
static nixlBackendPlugin plugin = {
    NIXL_PLUGIN_API_VERSION,
    create_posix_engine,
    destroy_posix_engine,
    get_plugin_name,
    get_plugin_version,
    get_backend_options,
    get_backend_mems
};

nixlBackendPlugin* createStaticPosixPlugin() {
    return &plugin; // Return the static plugin instance
}

#else

// Plugin initialization function
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin* nixl_plugin_init() {
    try {
        std::unique_ptr<nixlBackendPlugin> plugin = std::make_unique<nixlBackendPlugin>();
        plugin->create_engine = create_posix_engine;
        plugin->destroy_engine = destroy_posix_engine;
        plugin->get_plugin_name = get_plugin_name;
        plugin->get_plugin_version = get_plugin_version;
        plugin->get_backend_options = get_backend_options;
        plugin->get_backend_mems = get_backend_mems;
        plugin->api_version = NIXL_PLUGIN_API_VERSION;  // Set the API version
        return plugin.release();
    } catch (const std::exception& e) {
        return nullptr;
    }
}

// Plugin cleanup function
extern "C" NIXL_PLUGIN_EXPORT void nixl_plugin_fini() {
}

```

感谢您的阅读，今天的内容就到这里了。通过本文我们了解了c++项目NIXL插件框架的设计和实现，以及如何实现一个插件。