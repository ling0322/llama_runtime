#include <chrono>
#include <functional>

#include "test_helper.h"
#include "gemm.h"
#include "nn.h"
#include "operators.h"

using namespace llama;
using namespace nn;
using namespace std::literals;

void BenchmarkGEMM(int n, int num_run = 1) {
  auto F = Operators::FromDevice(Device::CPU());

  Tensor A = F->Rand({n, n}, DType::kFloat);
  Tensor B = F->Rand({n, n}, DType::kFloat);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_run; ++i) {
    Tensor C = F->MatMul(A, B);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  auto delta = t1 - t0;
  auto duration_ms = delta / 1ms / num_run;

  LOG(INFO) << "n = " << n << " t = " << duration_ms << "ms";
}

TEST_CASE("benchmark for float32 GEMM", "[gemm][benchmark]") {
  BenchmarkGEMM(1024, 10);
}
