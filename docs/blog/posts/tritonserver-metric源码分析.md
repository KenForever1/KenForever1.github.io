---
title: Tritonserver中metric使用源代码分析
date: 2025-11-29
authors: [KenForever1]
categories: 
  - LLM推理
labels: []
comments: true
---
对软件的监控是尤其重要的？不管是问题排查、性能优化。包括对metrics、logs、traces的监控。本文主要介绍tritonserver如何进行metrics数据收集。
<!-- more -->

> metrics数据的使用：
> 通过http服务 ( 例如：http://127.0.0.1:8002/metrics ）就可以获取到metrics数据。因为是符合prometheus的格式，所以可以直接对接prometheus。然后通过grafana可视化面板就可以监控和展示了。

## 源码实现一览

在tritonserver的core仓库中，包括了如下metrics的实现源码，本小节介绍一下是干什么的。
```c++
metrics.h/cpp: 核心类，metric_family和metric_reporter都使用了它，是全局的一个单例Metrics类。实现了metrics的注册、创建、删除、序列化等操作。也包括了gpu和cpu信息的收集。
metric_family.h/cpp: 是后面提到的为tritonserver自定义扩展metrics使用的，封装了counter和gauge类型的函数，比如Increment、Set、获取Value等函数。以及创建创建Family的函数。
metric_reporter.h/cpp: MetricModelReporter类是tritonserver内部调用的，用于收集metrics数据，比如推理成功了就会调用statistic相关函数，就会调用到MetricModelReporter类。
```

如果你也要为你的App实现metrics数据收集，就参考metrics.h/cpp就可以了。在tritonserver中代码比较独立。

## 如何收集metrics数据？

tritonserver通过prometheus的c++库(prometheus-cpp)来收集metrics数据。

简单看一下prometheus-cpp库的接口。

```c++
// 全局只有一个，创建的所有的family都会注册到这个registry中
std::shared_ptr<prometheus::Registry> registry_;
// 序列化器，用于将metrics数据序列化为字符串
std::unique_ptr<prometheus::Serializer> serializer_;

// 定义了几个Family对象，Family是prometheus-cpp库中定义的一个模板类
prometheus::Family<prometheus::Counter>& inf_success_family_;
prometheus::Family<prometheus::Gauge>& cache_num_entries_family_;
```

Family可以理解为prometheus-cpp库中定义的一个模板类，用于管理一组具有相同名称和帮助信息的指标。Family对象可以包含多个指标实例，每个实例都有不同的标签。比如：inf_success_family_可以统计模型1的请求成功次数，模型2的请求成功次数。

family如何创建：
```c++
inf_success_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_request_success")
              .Help("Number of successful inference requests, all batch sizes")
              .Register(*registry_)),
```

就为模型1和模型2分别创建Metric对象，然后Add到Family中。

```c++
// 添加一个labels的metric实例，family.Add(labels);
// 在释放时调用Remove函数移除。
prometheus::Counter*
MetricModelReporter::CreateCounterMetric(
    prometheus::Family<prometheus::Counter>& family,
    const std::map<std::string, std::string>& labels)
{
  return &family.Add(labels);
}


MetricModelReporter::~MetricModelReporter()
{
  Metrics::FamilyInferenceSuccess().Remove(metric_inf_success_);
}
```

Counter、Gauge、Histogram、Summary等都是Family的特化。用于不同统计场景的类型。Counter是计数器，Gauge是实时值，Histogram是直方图，Summary是摘要。

## 如何获取metrics数据？

通过启动一个http服务（tritonserver开源代码的src/http_server.cc中实现），调用prometheus-cpp库的接口序列化访问时的metrics数据，即得到一个metrics信息的string。prometheus可以解析使用了。

trtionserver采用的http库是evhtp，在业务上不一定使用这个库。
```c++
void
HTTPMetricsServer::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "HTTP request: " << req->method << " "
                 << req->uri->path->full;

  if (req->method != htp_method_GET) {
    RETURN_AND_RESPOND_WITH_ERR(
        req, EVHTP_RES_METHNALLOWED, "Method Not Allowed");
  }

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(kContentTypeHeader, "text/plain; charset=utf-8", 1, 1));

  // Call to metric endpoint should not have any trailing string
  if (RE2::FullMatch(std::string(req->uri->path->full), api_regex_)) {
    TRITONSERVER_Metrics* metrics = nullptr;
    TRITONSERVER_Error* err =
        TRITONSERVER_ServerMetrics(server_.get(), &metrics);
    if (err == nullptr) {
      const char* base;
      size_t byte_size;
      err = TRITONSERVER_MetricsFormatted(
          metrics, TRITONSERVER_METRIC_PROMETHEUS, &base, &byte_size);
      if (err == nullptr) {
        // 获取到metrics数据，直接返回给客户端
        evbuffer_add(req->buffer_out, base, byte_size);
      }
    }

    TRITONSERVER_MetricsDelete(metrics);
    RETURN_AND_RESPOND_IF_ERR(req, err);
    TRITONSERVER_ErrorDelete(err);
  }

  evhtp_send_reply(req, EVHTP_RES_OK);
}
```
TRITONSERVER_ServerMetrics是获取TRITONSERVER_Metrics对象（是一个c语言的空struct，是tritonserver将c++实现的类，导出c语言接口的一种方式）。这样实现FFI跨语言调用绑定，这些都很方便。

```c++

struct TRITONSERVER_Metrics;

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONSERVER_ServerMetrics(
    TRITONSERVER_Server* server, TRITONSERVER_Metrics** metrics)
{
#ifdef TRITON_ENABLE_METRICS
    // TritonServerMetrics才是内部真正使用的c++类，这个类只是为了调用单例类Metrics的序列化string函数
    // 这里new了，就要提供对应的c语言API的delete函数释放资源
  TritonServerMetrics* lmetrics = new TritonServerMetrics();
  *metrics = reinterpret_cast<TRITONSERVER_Metrics*>(lmetrics);
  return nullptr;  // Success
#else
  *metrics = nullptr;
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "metrics not supported");
#endif  // TRITON_ENABLE_METRICS
}
```

TRITONSERVER_MetricsFormatted函数就是序列化metrics数据，返回一个string。实现逻辑在tritonserver-core仓库中(src/tritonserver.cc)。

看一下proemtheus-cpp的提供的接口如何序列化metrics数据的，

```c++
const std::string
Metrics::SerializedMetrics()
{
  auto singleton = Metrics::GetSingleton();
  return singleton->serializer_->Serialize(
      singleton->registry_.get()->Collect());
}

class Metrics {
std::shared_ptr<prometheus::Registry> registry_;
std::unique_ptr<prometheus::Serializer> serializer_;
}
```

## tritonserver自定义metrics

tritonserver中默认已经定义了很多metrics，我们在开发backend的时候，可以通过trtionserver提供的接口，自定义metrics。

例子：[custom-metric-example](https://github.com/triton-inference-server/identity_backend/blob/main/README.md#custom-metric-example)

主要是用到了TRITONSERVER_MetricFamily* 和 TRITONSERVER_Metric* API接口的使用。

源码实现在core仓库中的metric_family.h/cpp中。

## 其他

+ [opentelemetry遥测数据收集](https://opentelemetry.io/docs/languages/cpp/)

opentelemetry-demo如何收集数据和grafana展示的架构图，可以参考这个文章[demo-architecture](https://opentelemetry.io/docs/demo/architecture/)。

除了prometheus，还使用了jaeger，支持分布式链路追踪，收集trace数据。

OpenTelemetry作为统一的可观测性框架，支持追踪（Traces）、指标（Metrics）、日志（Logs）三大支柱数据的采集，通过SDK嵌入应用代码或自动Instrumentation采集数据，并可灵活导出到多种后端系统（如Jaeger、Prometheus等。

如果只统计metrics数据，可以使用prometheus-cpp库更简单，依赖更少。如果需要分布式链路追踪，可以使用opentelemetry。

https://observability.cn/project/opentelemetry/by18dz6ns3tq2pcu/#_top


+ [mastering-prometheus-on-macos-seamless-integration-with-grafana](https://medium.com/@aravind33b/mastering-prometheus-on-macos-seamless-integration-with-grafana-5da2a1c95092)

结合上面这个简单的grafaan和prometheus的教程，可以快速搭建一个监控系统。在linux上可以通过nc开启一个metrics http服务。

```bash
nohup sh statistics.sh > log_metrics.txt 2>&1 &
```

```bash
# statistics.sh

#!/bin/bash
# 定义监控间隔（秒）
interval=10

# 获取系统总的CPU使用情况（sy: system, id: idle）
get_sys_cpu_usage() {
    top -bn1 | grep "%Cpu(s)" | awk '{print $4}'  # sy表示系统CPU占用
}

# 获取系统总的CPU空闲占用（id: idle）
get_total_cpu_usage() {
    top -bn1 | grep "%Cpu(s)" | awk '{print $8}'  # id表示空闲CPU占用
}

# 获取系统内存使用率（基于 MemAvailable）
get_memory_usage() {
    total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    available=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    used=$((total - available))
    usage=$(echo "scale=2; $used / $total * 100" | bc)
    echo "$usage"
}

# 定义监听的端口（默认 8080）
PORT=8121

# 启动 HTTP 服务函数
start_http_server() {
  echo "Starting HTTP metrics server on port $PORT..."
  while true; do
    # 系统资源
    sys_cpu=$(get_sys_cpu_usage)
    total_cpu=$(get_total_cpu_usage)
    # 计算total CPU使用率，total = 100 - id
    total_cpu_usage=$(awk "BEGIN {print 100 - $total_cpu}")
    # 获取系统内存使用率
    memory_usage=$(get_memory_usage)

    # 使用 nc (netcat) 监听 HTTP 请求
    { 
      printf "HTTP/1.1 200 OK\r\n";
      printf "Content-Type: text/plain\r\n";
      printf "\r\n";  # 空行分隔头和 body
      printf "sys_cpu $sys_cpu\n"
      printf "total_cpu $total_cpu\n"
      printf "total_cpu_usage $total_cpu_usage\n"
      printf "memory_usage $memory_usage\n"
    } | nc -l -p "$PORT" -w 1
  done
}

# 主逻辑
cleanup() {
  echo "Stopping metrics server..."
  rm -f "$METRICS_FILE"
  exit 0
}

trap cleanup EXIT

# 启动 HTTP 服务
start_http_server
```

jaeger:
https://www.liwenzhou.com/posts/Go/jaeger/