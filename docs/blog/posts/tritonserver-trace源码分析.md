---
title: Tritonserver中trace使用源代码分析
date: 2025-11-29
authors: [KenForever1]
categories: 
  - LLM推理
labels: []
comments: true
---

本文主要介绍了tritonserver中trace的实现，我们知道日志、metric和trace是常用的监控手段，可以用来分析问题、评估性能。tritonserver中的trace主要基于opentelemetry实现。

<!-- more -->

## trtionserver trace实现

### trace的两种模式

分为两种模式，triton模式就是将trace数据写入到文件，另一种opentelemetry模式就是将trace数据发送到opentelemetry collector(或者直接发送给jaeger查看，因为jaeger支持通过otlp协议通过http或者grpc接受trace数据)。

opentelemetry模式在tritonserver中采用opentelemetry-cpp库实现。

实现源码：

+ tritonserver项目中的src/tracer.h和src/tracer.cc
+ tritonserver的core实现中的src/infer_trace.h和src/infer_trace.cc

### tritonserver core中trace实现

core中的infer_trace主要实现了InferenceTrace类和InferenceTraceProxy类，在InferenceTrace类中没有直接使用opentelemetry-cpp库(没有对该库的依赖)，记录trace是使用的回调函数实现。

```c++
typedef void (*TRITONSERVER_InferenceTraceActivityFn_t)(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns,
    void* userp);

TRITONSERVER_InferenceTraceActivityFn_t activity_fn_;
TRITONSERVER_InferenceTraceTensorActivityFn_t tensor_activity_fn_;
TRITONSERVER_InferenceTraceReleaseFn_t release_fn_;
```

activity_fn_函数就是tritonserver的tracer.cc中实现的ReportToOpenTelemetry函数。

```c++
void
TraceManager::Trace::ReportToOpenTelemetry(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns)
{
  uint64_t id;
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceId(trace, &id), "getting trace id");
  if (span_stacks_.find(id) == span_stacks_.end()) {
    std::unique_ptr<
        std::stack<opentelemetry::nostd::shared_ptr<otel_trace_api::Span>>>
        st(new std::stack<
            opentelemetry::nostd::shared_ptr<otel_trace_api::Span>>());
    span_stacks_.emplace(id, std::move(st));
  }

  AddEvent(trace, activity, timestamp_ns, id);
}

void
TraceManager::TraceRelease(TRITONSERVER_InferenceTrace* trace, void* userp)
{
    // ......
}
```

InferenceTrace类和InferenceTraceProxy类主要是为了tritonserver的业务代码中方便使用，为trace保存了request、modelname、modelversion等信息，并提供了接口供业务代码使用。

所以在tritonserver的core实现中直接使用InferenceTrace类的接口来保存trace数据。

```c++
// core项目中src/infer_request.cc
#ifdef TRITON_ENABLE_STATS
void
InferenceRequest::ReportStatistics(
    MetricModelReporter* metric_reporter, bool success,
    const uint64_t compute_start_ns, const uint64_t compute_input_end_ns,
    const uint64_t compute_output_start_ns, const uint64_t compute_end_ns)
{
  if (!collect_stats_) {
    return;
  }

#ifdef TRITON_ENABLE_TRACING
  if (trace_ != nullptr) {
    trace_->Report(TRITONSERVER_TRACE_COMPUTE_START, compute_start_ns);
    trace_->Report(TRITONSERVER_TRACE_COMPUTE_INPUT_END, compute_input_end_ns);
    trace_->Report(
        TRITONSERVER_TRACE_COMPUTE_OUTPUT_START, compute_output_start_ns);
    trace_->Report(TRITONSERVER_TRACE_COMPUTE_END, compute_end_ns);
  }
#endif  // TRITON_ENABLE_TRACING

}
```

同时如果要自定义backend，也提供了API，在backend中记录trace。这些API函数就是调用了InferenceTrace类的接口。（r22.12版本中没有提供接口TRITONSERVER_InferenceTraceReportActivity,也没有实现opentelementry统计trace数据，在后面的版本提供了。）

```c++
backend trace的例子参考：https://github1s.com/triton-inference-server/identity_backend/blob/main/src/identity.cc
```

### tritonserver中opentelemetry实现

对opentelemtry-cpp库的使用，包括使用了http_collector_exporter、span记录等都在tritonserver项目中的tracer.cc中实现。

初始化tracker，

```c++
void
TraceManager::InitTracer(const triton::server::TraceConfigMap& config_map)
{
  switch (global_setting_->mode_) {
    case TRACE_MODE_OPENTELEMETRY: {
#ifndef _WIN32
      otlp::OtlpHttpExporterOptions exporter_options;
      otel_resource::ResourceAttributes attributes = {};
      otel_trace_sdk::BatchSpanProcessorOptions processor_options;

      ProcessOpenTelemetryParameters(
          config_map, exporter_options, attributes, processor_options);

      auto exporter = otlp::OtlpHttpExporterFactory::Create(exporter_options);
      auto processor = otel_trace_sdk::BatchSpanProcessorFactory::Create(
          std::move(exporter), processor_options);
      auto resource = otel_resource::Resource::Create(attributes);
      std::shared_ptr<otel_trace_api::TracerProvider> provider =
          otel_trace_sdk::TracerProviderFactory::Create(
              std::move(processor), resource);

      otel_trace_api::Provider::SetTracerProvider(provider);
      otel_cntxt::propagation::GlobalTextMapPropagator::SetGlobalPropagator(
          opentelemetry::nostd::shared_ptr<
              otel_cntxt::propagation::TextMapPropagator>(
              new otel_trace_api::propagation::HttpTraceContext()));
      break;
#else
      LOG_ERROR << "Unsupported trace mode: "
                << TraceManager::InferenceTraceModeString(
                       global_setting_->mode_);
      break;
#endif
    }
    default:
      return;
  }
}
```

记录trace的核心函数是ReportToOpenTelemetry, 在其中会调用StartSpan函数和EndSpan函数。

```c++
void
TraceManager::Trace::ReportToOpenTelemetry(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns){}

```

在tritonserver中通过TRITONSERVER_InferenceTraceActivity的字符串后缀来区分是startspan还是endspan。

```c++
static std::string start = "_START";
static std::string end = "_END";

if (activity == TRITONSERVER_TRACE_REQUEST_START ||
    activity == TRITONSERVER_TRACE_COMPUTE_START ||
    (activity == TRITONSERVER_TRACE_CUSTOM_ACTIVITY &&
    activity_name.length() > start.length() &&
    std::equal(start.rbegin(), start.rend(), activity_name.rbegin()))) {
    StartSpan(trace, activity, timestamp_ns, trace_id, span_name);
}
if (activity == TRITONSERVER_TRACE_REQUEST_END ||
    activity == TRITONSERVER_TRACE_COMPUTE_END ||
    (activity == TRITONSERVER_TRACE_CUSTOM_ACTIVITY &&
    activity_name.length() > end.length() &&
    std::equal(end.rbegin(), end.rend(), activity_name.rbegin()))) {
    EndSpan(timestamp_ns, trace_id);
}

```

system_clock‌：适用于需要与实际日历时间关联的场景（如日志记录、显示当前时间），支持与time_t类型相互转换
‌steady_clock‌：专为高精度计时设计，如性能分析或算法耗时测量，但无法直接转换为日历时间

## opentelemetry-cpp库的使用

[GettingStarted](https://opentelemetry-cpp.readthedocs.io/en/latest/sdk/GettingStarted.html)

+ 可以通过Context Propagation（上下文传播）跨进程传递Trace上下文信息，在C++中主要依赖W3C TraceContext标准的HTTP头部或gRPC元数据实现

## jeager配置使用，可视化trace数据

```c++
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:1.49
```

## opentelemetry-collector配置使用

在新版本中opentelementry删除了jeager协议的exporter，使用otlp/jeager。

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318
exporters:
  # NOTE: Prior to v0.86.0 use `logging` instead of `debug`.
  debug:
    verbosity: detailed
  otlp/jaeger:
    # Jaeger supports OTLP directly
    endpoint: localhost:4317
    tls:
      insecure: true
processors:
  batch:
service:
  pipelines:
    traces:
      receivers: [ otlp ]
      exporters: [ otlp/jaeger ]
      processors: [ batch ]
    metrics:
      receivers: [ otlp ]
      exporters: [ debug ]
      processors: [ batch ]
    logs:
      receivers: [ otlp ]
      exporters: [ debug ]
      processors: [ batch ]

```

```bash
docker run -p 4317:4317 \
    -v $(pwd)/otel-collector-config.yaml:/etc/otel-collector-config.yaml \
    otel/opentelemetry-collector:latest \
    --config=/etc/otel-collector-config.yaml
```

## 参考

<https://triton-inference-server.github.io/pytriton/0.5.2/guides/distributed_tracing/>

<https://github1s.com/triton-inference-server/server/blob/main/src/tracer.h#L41>
