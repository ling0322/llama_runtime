#include <chrono>
#include <functional>

#include "cblas.h"
#include "test_helper.h"
#include "gemm.h"
#include "nn.h"
#include "operators.h"

using namespace llama;
using namespace nn;
using namespace std::literals;

Tensor CallGEMM(Operators *F, TensorCRef A, TensorCRef B) {
  return F->MatMul(A, B);
}

Tensor CallOpenblasGEMM(Operators *F, TensorCRef A, TensorCRef B) {
  const float *A_data;
  Tensor C = F->Tensor_({A.shape(0), B.shape(1)}, DType::kFloat);
  cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      A.shape(0), B.shape(1), A.shape(0),
      1.0f,
      A.data<float>(), A.stride(0),
      B.data<float>(), B.stride(0),
      0.0f,
      C.data<float>(), C.stride(0));

  return C;
}

enum GemmType {
  kFastAlpaca,
  kOpenblas
};

void BenchmarkGEMM(int n, GemmType gemm_type, int num_run = 1) {
  auto F = Operators::FromDevice(Device::CPU());

  Tensor A = F->Rand({n, n}, DType::kFloat);
  Tensor B = F->Rand({n, n}, DType::kFloat);

  Tensor C = CallGEMM(F.get(), A, B);
  Tensor C_openblas = CallOpenblasGEMM(F.get(), A, B);
  F->Print(C);
  F->Print(C_openblas);
  // REQUIRE(F->AllClose(C, C_openblas));

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_run; ++i) {
    Tensor C;
    switch (gemm_type) {
      case kFastAlpaca:
        C = CallGEMM(F.get(), A, B);
        break;
      case kOpenblas:
        C = CallOpenblasGEMM(F.get(), A, B);
        break;
      default:
        NOT_IMPL();
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  auto delta = t1 - t0;
  auto duration_ms = delta / 1ms / num_run;

  LOG(INFO) << "n = " << n << " t = " << duration_ms << "ms";
}

TEST_CASE("benchmark for float32 GEMM", "[gemm][benchmark]") {
  openblas_set_num_threads(1);
  BenchmarkGEMM(1024, kFastAlpaca, 100);
  BenchmarkGEMM(1024, kOpenblas, 100);
  BenchmarkGEMM(2048, kFastAlpaca, 10);
  BenchmarkGEMM(2048, kOpenblas, 10);
  BenchmarkGEMM(4096, kFastAlpaca, 5);
  BenchmarkGEMM(4096, kOpenblas, 5);
}
