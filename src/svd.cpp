#include <arrayfire.h>
#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>

#include <cmath>
#include <vector>

using af::array;
using af::deviceGC;
using af::deviceMemInfo;
using af::dim4;
using af::svd;
using std::vector;

static void svdBase(benchmark::State &state, dim_t edgeLength, af_dtype type) {
  dim4 dims(edgeLength, edgeLength);
  array in = array(dims, type);
  {
    array u;
    array s_vec;
    array vt;
    svd(u, s_vec, vt, in);
    af::sync();
  }

  size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

  for (auto _ : state) {
    array u;
    array s_vec;
    array vt;
    svd(u, s_vec, vt, in);
    u.eval();
    s_vec.eval();
    vt.eval();
    af::sync();
  }

  size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

  state.counters["bytes"] = (alloc_bytes2 - alloc_bytes);

  double ops = pow(edgeLength, 3);
  state.counters["flops"] =
      benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate);

  deviceGC();
}

int main(int argc, char **argv) {
  using af::benchmark::RegisterBenchmark;

  const af_backend backend = (argc > 1 ? AF_BACKEND_CPU : AF_BACKEND_DEFAULT);
  af::setBackend(backend);

  vector<af_dtype> types = {f32};

  RegisterBenchmark("SVD", types,
                    [](benchmark::State &state, af_dtype type) {
                      svdBase(state, state.range(0), type);
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
