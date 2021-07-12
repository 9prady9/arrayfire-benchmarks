#include <arrayfire.h>
#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>

#include <cmath>
#include <vector>

using af::array;
using af::convolve;
using af::deviceGC;
using af::deviceMemInfo;
using af::dim4;
using af::randu;
using std::vector;

static void convBase(benchmark::State &state, dim_t edgeLength,
                     unsigned kernelEdgeLength, af_dtype type) {
  array in = randu(dim4(edgeLength, edgeLength), type);
  array k0 = randu(dim4(kernelEdgeLength), type);
  array k1 = randu(dim4(1, kernelEdgeLength), type);
  {
    array out = convolve(k0, k1, in);
    af::sync();
  }
  size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

  for (auto _ : state) {
    array out = convolve(k0, k1, in);
    af::sync();
  }
  size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

  state.counters["bytes"] = (alloc_bytes2 - alloc_bytes);

  double ops = edgeLength * edgeLength * (8 * kernelEdgeLength - 1);
  state.counters["flops"] =
      benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate);

  deviceGC();
}

int main(int argc, char **argv) {
  using af::benchmark::RegisterBenchmark;

  const af_backend backend = (argc > 1 ? AF_BACKEND_CPU : AF_BACKEND_DEFAULT);
  af::setBackend(backend);

  vector<af_dtype> types = {f32};

  RegisterBenchmark("separable_convolve", types,
                    [](benchmark::State &state, af_dtype type) {
                      convBase(state, state.range(0), 7, type);
                    })
      ->DenseRange(256, 4096, 256)
      ->UseRealTime()
      ->ArgNames({"EdgeLength"})
      ->Unit(benchmark::kMicrosecond);

  benchmark::Initialize(&argc, argv);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);

  return 0;
}
