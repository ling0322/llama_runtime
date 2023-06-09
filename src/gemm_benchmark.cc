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

Tensor callGEMM(Operators *F, TensorCRef A, TensorCRef B) {
  return F->matmul(A, B);
}

Tensor callOpenblasGEMM(Operators *F, TensorCRef A, TensorCRef B) {
  Tensor C = F->createTensor({A.getShape(0), B.getShape(1)}, DType::kFloat);
  cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      A.getShape(0), B.getShape(1), A.getShape(0),
      1.0f,
      A.getData<float>(), A.getStride(0),
      B.getData<float>(), B.getStride(0),
      0.0f,
      C.getData<float>(), C.getStride(0));

  return C;
}

enum GemmType {
  kFastAlpaca,
  kOpenblas
};

void benchmarkGEMM(int n, GemmType gemm_type, int num_run = 1) {
  auto F = Operators::create(Device::createForCPU());

  Tensor A = F->rand({n, n}, DType::kFloat);
  Tensor B = F->rand({n, n}, DType::kFloat);

  Tensor C = callGEMM(F.get(), A, B);
  Tensor C_openblas = callOpenblasGEMM(F.get(), A, B);
  F->print(C);
  F->print(C_openblas);
  // REQUIRE(F->AllClose(C, C_openblas));

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_run; ++i) {
    Tensor C;
    switch (gemm_type) {
      case kFastAlpaca:
        C = callGEMM(F.get(), A, B);
        break;
      case kOpenblas:
        C = callOpenblasGEMM(F.get(), A, B);
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
  benchmarkGEMM(1024, kOpenblas, 100);
  benchmarkGEMM(1024, kFastAlpaca, 100);
  benchmarkGEMM(2048, kFastAlpaca, 10);
  benchmarkGEMM(2048, kOpenblas, 10);
  benchmarkGEMM(4096, kFastAlpaca, 5);
  benchmarkGEMM(4096, kOpenblas, 5);
}
