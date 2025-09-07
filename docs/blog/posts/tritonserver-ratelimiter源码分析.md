---
title: tritonserver-ratelimiter源码分析
date: 2025-06-21
authors: [KenForever1]
categories: 
  - LLM推理
labels: []
comments: true
---

<!-- more -->
## tritonserver调度器

三种调度器：

+ emsemble schedulor

+ dynamicbatch scheduler

+ sequece scheduler

以metric为入口对源码进行分析，

metric提供了http接口，统计了nv_inference_compute_infer_duration_us和nv_inference_queue_duration_us指标。

```python
count_pattern = re.compile(r'nv_inference_count\{model="([^"]+)".*\} (\d+)')
duration_pattern = re.compile(r'nv_inference_request_duration_us\{model="([^"]+)".*\} (\d+)')
compute_duration_pattern = re.compile(r'nv_inference_compute_infer_duration_us\{model="([^"]+)".*\} (\d+)')
queue_duration_pattern = re.compile(r'nv_inference_queue_duration_us\{model="([^"]+)".*\} (\d+)')
```
## metric统计nv_inference_queue_duration_us和nv_inference_compute_infer_duration_us指标

注册指标到prometheus_registry_中。
```c++
      inf_request_duration_us_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_request_duration_us")
              .Help("Cumulative inference request duration in microseconds "
                    "(includes cached requests)")
              .Register(*registry_)),
      inf_queue_duration_us_family_(
          prometheus::BuildCounter()
              .Name("nv_inference_queue_duration_us")
              .Help("Cumulative inference queuing duration in microseconds "
                    "(includes cached requests)")
              .Register(*registry_)),
```

```c++
counter_families_["queue_duration"] =
        &Metrics::FamilyInferenceQueueDuration();
```

全局搜索“queue_duration”找到“uint64_t queue_duration_ns_;”在infer_stats.h中。进而可以找到如何统计的了。
```c++
const uint64_t queue_duration_ns = compute_start_ns - queue_start_ns;
```

## Dynamicbatch scheduler中调度流程

RateLimiter::EnqueuePayload：DynamicBatchScheduler::BatcherThread线程调度。cv_.wait_for，可以通过查询cv_找到在哪儿放入资源的，也就是cv_.notify_one()。DynamicBatchScheduler::Enqueue函数中，将requests Enqueue。

前面说的metric统计的入队时间就是在DynamicBatchScheduler::Enqueue函数函数进入时初始化的， request->CaptureQueueStartNs()。(Emsemble调度中也有这个Enqueue操作，初始化enqueue时间点)。

SchedulePayload：将playload放入payload_queue->queue_和payload_queue->specific_queues_（与具体的model_instance相关）。

RateLimiter::DequeuePayload：TritonModelInstance::TritonBackendThread::BackendThread()模型实例线程调用，由从payload_queue->queue_->Dequeue获取payload，或者对playload中的requests进行**merge（batch调度）**。

RateLimiter中的merge会根据配置文件中的**最大batch数和等待时间**进行merge。

## Esemble scheduler中调度流程

```c++
void
EnsembleContext::Proceed(
    const std::shared_ptr<EnsembleContext>& context,
    const std::unique_ptr<Step>& completed_step)
{
  StepList ready_steps;
  Status status = context->PrepareSteps(completed_step, &ready_steps);
  if (status.IsOk()) {
    ScheduleSteps(context, std::move(ready_steps));
  }
}
```