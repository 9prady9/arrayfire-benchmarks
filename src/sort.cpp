#include <arrayfire.h>
#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>

#include <vector>

using af::array;
using af::deviceGC;
using af::deviceMemInfo;
using af::dim4;
using af::randu;
using af::sum;
using std::vector;

void sortBench(benchmark::State &state, af_dtype type) {
  af::dim4 dataDims(state.range(0), state.range(1));
  unsigned sortDim = state.range(2);
  array input = randu(dataDims, type);
  array output = sort(input, sortDim);
  af::sync();

  size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

  for (auto _ : state) {
    output = sort(input, sortDim);
    af::sync();
  }

  size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

  state.counters["Bytes"] = (alloc_bytes2 - alloc_bytes);

  state.counters["ElementsPerSecond"] = benchmark::Counter(
      dataDims.elements(), benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::OneK::kIs1024);
  deviceGC();
}

void sortBench2(benchmark::State &state, dim_t length, af_dtype type) {
  array in = randu(dim4(length), type);
  array out = sort(in, 0);
  af::sync();
  size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

  for (auto _ : state) {
    out = sort(in, 0);
    af::sync();
  }
  size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

  state.counters["Bytes"] = (alloc_bytes2 - alloc_bytes);

  state.counters["ElementsPerSecond"] =
      benchmark::Counter(length, benchmark::Counter::kIsIterationInvariantRate);
  deviceGC();
}

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);

  const bool websiteBench = (argc > 1 ? (atoi(argv[1]) > 0) : false);

  const af_backend backend = (argc > 2 ? AF_BACKEND_CPU : AF_BACKEND_DEFAULT);
  af::setBackend(backend);

  vector<af_dtype> types = {f32};
  if (!websiteBench) {
    types.emplace_back(f64);
  }

  af::benchmark::RegisterBenchmark("sort", types,
                                   [&](benchmark::State &state, af_dtype type) {
                                     sortBench2(state, state.range(0), type);
                                   })
      ->DenseRange(1000000, 20000000, 1000000)
      ->UseRealTime()
      ->ArgNames({"EdgeLength"})
      ->Unit(benchmark::kMicrosecond);

  if (!websiteBench) {
    af::benchmark::RegisterBenchmark("sort", types, sortBench)
        ->Ranges({{8, 1 << 12}, {8, 1 << 12}, {0, 1}})
        ->UseRealTime()
        ->ArgNames({"dim0", "dim1", "sortDim"})
        ->Unit(benchmark::kMicrosecond);
  }

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
