#include <arrayfire.h>
#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>

#include <cmath>
#include <vector>

using af::array;
using af::deviceGC;
using af::deviceMemInfo;
using af::dim4;
using af::randu;
using af::transpose;
using std::vector;

static void transposeBase(benchmark::State &state, dim4 dims, af_dtype type) {
  array in = randu(dims, type);
  {
    array out = transpose(in);
    af::sync();
  }
  size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

  for (auto _ : state) {
    array out = transpose(in);
    af::sync();
  }
  size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

  state.counters["bytes"] = (alloc_bytes2 - alloc_bytes);

  state.counters["flops"] = benchmark::Counter(
      2.0 * dims.elements(), benchmark::Counter::kIsIterationInvariantRate);

  deviceGC();
}

int main(int argc, char **argv) {
  using af::benchmark::RegisterBenchmark;

  const bool websiteBench = (argc > 1 ? (atoi(argv[1]) > 0) : false);

  const af_backend backend = (argc > 2 ? AF_BACKEND_CPU : AF_BACKEND_DEFAULT);
  af::setBackend(backend);

  vector<af_dtype> types = {f32};

  RegisterBenchmark("transpose", types,
                    [](benchmark::State &state, af_dtype type) {
                      unsigned E = state.range(0);
                      af::dim4 dims(E, E);
                      transposeBase(state, dims, type);
                    })
      ->DenseRange(256, 4096, 256)
      ->UseRealTime()
      ->ArgNames({"EdgeLength"})
      ->Unit(benchmark::kMicrosecond);

  if (!websiteBench) {
    RegisterBenchmark("transposeTallColumn", types,
                      [](benchmark::State &state, af_dtype type) {
                        unsigned M = state.range(0);
                        unsigned N = state.range(1);
                        af::dim4 dims(M, N);
                        transposeBase(state, dims, type);
                      })
        ->RangeMultiplier(2)
        ->Ranges({{64, 64}, {64, 1 << 16}})
        ->UseRealTime()
        ->ArgNames({"M", "N"})
        ->Unit(benchmark::kMicrosecond);

    RegisterBenchmark("transposeTallRow", types,
                      [](benchmark::State &state, af_dtype type) {
                        unsigned M = state.range(0);
                        unsigned N = state.range(1);
                        af::dim4 dims(M, N);
                        transposeBase(state, dims, type);
                      })
        ->RangeMultiplier(2)
        ->Ranges({{64, 64}, {64, 1 << 16}})
        ->UseRealTime()
        ->ArgNames({"M", "N"})
        ->Unit(benchmark::kMicrosecond);
  }
  benchmark::Initialize(&argc, argv);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);

  return 0;
}
