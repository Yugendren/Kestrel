#include "common/perf/frameTimeStats.h"
#include "common/perf/hostStats.h"
#include "common/perf/perfSampler.h"
#include "common/profiler.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using Common::Perf::FrameTimeHistory;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "PerfStatsTests: failed: %s\n", message);
    std::abort();
  }
}

bool Near(double value, double expected, double tolerance) {
  return std::fabs(value - expected) <= tolerance;
}

void TestSummarizeEmpty() {
  const auto summary = Common::Perf::SummarizeFrameTimes({});
  Check(summary.frames == 0 && summary.fps == 0.0 && summary.avg_ms == 0.0 &&
            summary.last_ms == 0.0 && summary.max_ms == 0.0 &&
            summary.low1_fps == 0.0,
        "empty sequence is not all zeros");
}

void TestSummarizeConstant() {
  const std::vector<float> frames(120, 16.6667f);
  const auto summary = Common::Perf::SummarizeFrameTimes(frames);
  Check(summary.frames == 120, "constant: frame count");
  Check(Near(summary.fps, 60.0, 0.01), "constant: fps is not 60");
  Check(Near(summary.avg_ms, 16.6667, 0.001), "constant: avg");
  Check(Near(summary.low1_fps, summary.fps, 0.01),
        "constant: 1% low differs from fps");
}

void TestSummarizeSpike() {
  // One 100 ms hitch in 200 frames: nearest-rank p99 is the 198th smallest
  // frame, still a normal one, so the 1% low stays at 60 fps.
  std::vector<float> frames(200, 16.6667f);
  frames[100] = 100.0f;
  auto summary = Common::Perf::SummarizeFrameTimes(frames);
  Check(summary.frames == 200, "spike: frame count");
  Check(Near(summary.max_ms, 100.0, 1e-6), "spike: max");
  Check(Near(summary.last_ms, 16.6667, 1e-4), "spike: last");
  Check(Near(summary.avg_ms, (199 * 16.6667 + 100.0) / 200.0, 1e-3),
        "spike: avg");
  Check(Near(summary.fps, 1000.0 / summary.avg_ms, 1e-6), "spike: fps");
  Check(Near(summary.low1_fps, 60.0, 0.01), "spike: 1% low in 200 frames");

  // In 50 frames the p99 rank is 50, i.e. the hitch itself.
  frames.assign(50, 16.6667f);
  frames[10] = 100.0f;
  summary = Common::Perf::SummarizeFrameTimes(frames);
  Check(Near(summary.low1_fps, 10.0, 1e-6), "spike: 1% low in 50 frames");
}

void TestHistoryWindowEviction() {
  FrameTimeHistory history(1'000'000);
  Check(history.LastPresentUs() == 0, "empty history has a last present");
  for (uint64_t t = 0; t <= 2'000'000; t += 10'000) {
    history.RecordPresent(t);
  }
  Check(history.LastPresentUs() == 2'000'000, "last present");

  std::vector<float> out;
  // Timestamps 1.0 s .. 2.0 s are retained: 101 presents, 100 frame times.
  history.CopyFrameTimes(2'000'000, 10'000'000, &out);
  Check(out.size() == 100, "window eviction kept the wrong number of frames");
  for (const float ms : out) {
    Check(Near(ms, 10.0, 1e-4), "window eviction: frame time");
  }

  // (1.5 s, 2.0 s] holds 50 presents.
  history.CopyFrameTimes(2'000'000, 500'000, &out);
  Check(out.size() == 50, "span filter kept the wrong number of frames");
  // Presents after `now` are excluded.
  history.CopyFrameTimes(1'900'000, 500'000, &out);
  Check(out.size() == 50, "span filter included future presents");
}

void TestHistoryCopyLatest() {
  FrameTimeHistory history(100'000);
  std::array<float, 8> out{};
  Check(history.CopyLatest(out) == 0, "empty history copied frames");

  history.RecordPresent(1'000);
  history.RecordPresent(3'000);
  history.RecordPresent(6'000);
  Check(history.CopyLatest(out) == 2, "partial fill count");
  Check(Near(out[0], 2.0, 1e-6) && Near(out[1], 3.0, 1e-6),
        "partial fill values");

  // 3000 presents with a 100 ms window never outgrow the initial ring, so
  // the ring wraps around many times. Intervals cycle 1, 2, 3 ms.
  FrameTimeHistory wrapping(100'000);
  uint64_t t = 0;
  std::vector<float> expected;
  for (int i = 0; i < 3000; i++) {
    const uint64_t step = 1'000 * static_cast<uint64_t>(i % 3 + 1);
    t += step;
    wrapping.RecordPresent(t);
    expected.push_back(static_cast<float>(step) / 1000.0f);
  }
  Check(wrapping.CopyLatest(out) == out.size(), "wraparound count");
  for (size_t i = 0; i < out.size(); i++) {
    Check(Near(out[i], expected[expected.size() - out.size() + i], 1e-6),
          "wraparound values out of order");
  }
}

void TestHistoryGrowth() {
  // 5000 presents 1 ms apart all fit a 10 s window: the ring must grow past
  // its initial capacity without losing or reordering frames.
  FrameTimeHistory history(10'000'000);
  for (uint64_t i = 1; i <= 5000; i++) {
    history.RecordPresent(i * 1'000);
  }
  std::vector<float> out;
  history.CopyFrameTimes(5'000'000, 10'000'000, &out);
  Check(out.size() == 4999, "growth lost frames");
  for (const float ms : out) {
    Check(Near(ms, 1.0, 1e-6), "growth: frame time");
  }
  std::array<float, 3> latest{};
  Check(history.CopyLatest(latest) == 3, "growth: latest count");
}

void TestCpuUsageSampler() {
  std::atomic<bool> ready{false};
  std::atomic<bool> stop{false};
  std::thread spinner([&] {
    Profiler::SetThreadName("Thread_Gpu");
    ready = true;
    volatile uint64_t sink = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      sink = sink + 1;
    }
  });
  while (!ready) {
    std::this_thread::yield();
  }

  Common::Perf::CpuUsageSampler sampler;
  const auto primed = sampler.Sample();
  Check(primed.process_core_percent == 0 && !primed.gpu_thread.has_value(),
        "first sample is not zero");
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  const auto usage = sampler.Sample();
  stop = true;
  spinner.join();

  std::printf("cpu: process %.1f%% (system share %.1f%%), system %.1f%%, "
              "busiest %s %.1f%%\n",
              usage.process_core_percent, usage.process_system_percent,
              usage.system_percent, usage.busiest.name.c_str(),
              usage.busiest.core_percent);
  Check(usage.gpu_thread.has_value(), "Thread_Gpu not found");
  std::printf("cpu: Thread_Gpu tid %llu %.1f%%\n",
              static_cast<unsigned long long>(usage.gpu_thread->id),
              usage.gpu_thread->core_percent);
  Check(usage.gpu_thread->core_percent > 70, "Thread_Gpu usage too low");
  Check(usage.busiest.id == usage.gpu_thread->id,
        "busiest thread is not the spinner");
  Check(usage.process_core_percent >= 70, "process usage too low");
  Check(usage.process_system_percent <= 100 && usage.system_percent <= 100,
        "percentages out of range");
}

void TestMemoryUsage() {
  const auto memory = Common::Perf::SampleMemoryUsage();
  std::printf("memory: used %llu / %llu MiB, process %llu MiB\n",
              static_cast<unsigned long long>(memory.system_used_bytes >> 20),
              static_cast<unsigned long long>(memory.system_total_bytes >> 20),
              static_cast<unsigned long long>(memory.process_bytes >> 20));
  Check(memory.system_used_bytes > 0, "no system memory used");
  Check(memory.system_used_bytes <= memory.system_total_bytes,
        "used exceeds total");
  Check(memory.process_bytes > 0, "no process memory");
}

void TestHostSpecs() {
  const auto specs = Common::Perf::CollectHostSpecs({});
  std::printf("host: %s, %u cores / %u threads, %llu MiB, %s\n",
              specs.cpu_model.c_str(), specs.physical_cores,
              specs.logical_threads,
              static_cast<unsigned long long>(specs.ram_bytes >> 20),
              specs.os.c_str());
  Check(!specs.cpu_model.empty() && !specs.os.empty(), "missing host names");
  Check(specs.physical_cores > 0 &&
            specs.logical_threads >= specs.physical_cores,
        "core counts");
  Check(specs.ram_bytes > 0, "no RAM");
}

void TestNvml() {
  auto gpu = Common::Perf::NvmlGpuMonitor::Open("");
  if (gpu == nullptr) {
    std::puts("gpu: NVML unavailable");
    return;
  }
  const auto usage = gpu->Sample();
  Check(usage.has_value(), "NVML sample failed");
  std::printf("gpu: util %u%%, vram %llu / %llu MiB, temp %d C, power %d W\n",
              usage->utilization_percent,
              static_cast<unsigned long long>(usage->vram_used_bytes >> 20),
              static_cast<unsigned long long>(usage->vram_total_bytes >> 20),
              usage->temperature_c ? static_cast<int>(*usage->temperature_c)
                                   : -1,
              usage->power_w ? static_cast<int>(*usage->power_w) : -1);
  Check(usage->utilization_percent <= 100, "GPU utilization above 100");
  Check(usage->vram_total_bytes > 0, "no VRAM");
}

void TestPerfSampler() {
  FrameTimeHistory history;
  const auto now_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
  for (uint64_t i = 60; i > 0; i--) {
    history.RecordPresent(now_us - i * 16'667);
  }

  Common::Perf::PerfSampler sampler(history, nullptr, 50);
  Check(sampler.Latest().sequence == 0, "inactive sampler published");
  sampler.SetActive(true);
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  const auto snapshot = sampler.Latest();
  sampler.SetActive(false);
  std::printf("sampler: seq %llu, last second %u frames %.1f fps, "
              "cost %.2f ms\n",
              static_cast<unsigned long long>(snapshot.sequence),
              snapshot.last_second.frames, snapshot.last_second.fps,
              snapshot.sample_cost_ms);
  Check(snapshot.sequence > 0, "active sampler did not publish");
  Check(snapshot.window.frames == 59, "sampler window frame count");
  Check(!snapshot.gpu.has_value(), "gpu without a monitor");
}

} // namespace

int main() {
  TestSummarizeEmpty();
  TestSummarizeConstant();
  TestSummarizeSpike();
  TestHistoryWindowEviction();
  TestHistoryCopyLatest();
  TestHistoryGrowth();
  TestCpuUsageSampler();
  TestMemoryUsage();
  TestHostSpecs();
  TestNvml();
  TestPerfSampler();
  std::puts("PerfStatsTests: all cases passed");
  return 0;
}
