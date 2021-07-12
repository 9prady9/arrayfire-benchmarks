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
using af::scan;
using std::vector;

static void scanBase(benchmark::State &state, dim_t numElems, af_dtype type) {
  array in = randu(dim4(numElems), type);
  {
    array out = scan(in);
    af::sync();
  }
  size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

  for (auto _ : state) {
    array out = scan(in);
    af::sync();
  }
  size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

  state.counters["bytes"] = (alloc_bytes2 - alloc_bytes);


  state.counters["flops"] = benchmark::Counter(
      2.0 * numElems, benchmark::Counter::kIsIterationInvariantRate);

  deviceGC();
}

int main(int argc, char **argv) {
  using af::benchmark::RegisterBenchmark;

  const af_backend backend = (argc > 1 ? AF_BACKEND_CPU : AF_BACKEND_DEFAULT);
  af::setBackend(backend);

  vector<af_dtype> types = {f32};

  RegisterBenchmark("scan", types,
                    [](benchmark::State &state, af_dtype type) {
                      scanBase(state, state.range(0), type);
                    })
      ->DenseRange(100000, 2000000, 100000)
      ->UseRealTime()
      ->ArgNames({"EdgeLength"})
      ->Unit(benchmark::kMicrosecond);

  benchmark::Initialize(&argc, argv);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);

  return 0;
}
