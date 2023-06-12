#include <chrono>
#include <functional>

#include "cblas.h"
#include "test_helper.h"
#include "gemm.h"
#include "nn.h"
#include "operators.h"
#include "strings.h"

using namespace llama;
using namespace nn;
using namespace std::literals;

Tensor callGEMM(Operators *F, TensorCRef A, TensorCRef B) {
  return F->matmul(A, B);
}

Tensor callGEMV(Operators *F, TensorCRef A, TensorCRef B) {
  return F->gemv(A, B);
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

Tensor callOpenblasGEMV(Operators *F, TensorCRef A, TensorCRef B) {
  Tensor C = F->createTensor({A.getShape(1)}, DType::kFloat);
  cblas_sgemv(
      CblasRowMajor, CblasNoTrans, A.getShape(0), A.getShape(1), 1.0f, A.getData<float>(),
      A.getStride(0), B.getData<float>(), 1, 0.0f, C.getData<float>(), 1);

  return C;
}

enum GEMMType {
  GEMM_LLMRT,
  GEMM_OPENBLAS,
  GEMV_LLMRT,
  GEMV_OPENBLAS
};


void benchmarkGEMM(int n, GEMMType gemmType, int numRun = 1) {
  auto F = Operators::create(Device::createForCPU());

  Tensor A = F->rand({n, n}, DType::kFloat);
  Tensor B = gemmType == GEMM_LLMRT || gemmType == GEMM_OPENBLAS
      ? F->rand({n, n}, DType::kFloat)
      : F->rand({n}, DType::kFloat);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < numRun; ++i) {
    Tensor C;
    switch (gemmType) {
      case GEMM_LLMRT:
        C = callGEMM(F.get(), A, B);
        break;
      case GEMM_OPENBLAS:
        C = callOpenblasGEMM(F.get(), A, B);
        break;
      case GEMV_LLMRT:
        C = callGEMV(F.get(), A, B);
        break;
      case GEMV_OPENBLAS:
        C = callOpenblasGEMV(F.get(), A, B);
        break;
      default:
        NOT_IMPL();
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  auto delta = t1 - t0;
  auto duration_ms = delta / 1ns / numRun / 1e6f;

  LOG(INFO) << str::sprintf("n = %d t = %.2f ms", n, duration_ms);
}

TEST_CASE("benchmark for float32 GEMM", "[gemm][benchmark]") {
  openblas_set_num_threads(1);
  LOG(INFO) << "openblas SGEMM:";
  benchmarkGEMM(256, GEMM_OPENBLAS, 400);
  benchmarkGEMM(512, GEMM_OPENBLAS, 200);
  benchmarkGEMM(1024, GEMM_OPENBLAS, 50);
  benchmarkGEMM(2048, GEMM_OPENBLAS, 5);
  benchmarkGEMM(4096, GEMM_OPENBLAS, 1);
  LOG(INFO) << "FastAlpaca SGEMM:";
  benchmarkGEMM(256, GEMM_LLMRT, 400);
  benchmarkGEMM(512, GEMM_LLMRT, 200);
  benchmarkGEMM(1024, GEMM_LLMRT, 50);
  benchmarkGEMM(2048, GEMM_LLMRT, 5);
  benchmarkGEMM(4096, GEMM_LLMRT, 1);
}

TEST_CASE("benchmark for float32 GEMV", "[gemv][benchmark]") {
  openblas_set_num_threads(1);
  LOG(INFO) << "openblas SGEMV:";
  benchmarkGEMM(256, GEMV_OPENBLAS, 400);
  benchmarkGEMM(512, GEMV_OPENBLAS, 200);
  benchmarkGEMM(1024, GEMV_OPENBLAS, 50);
  benchmarkGEMM(2048, GEMV_OPENBLAS, 5);
  benchmarkGEMM(4096, GEMV_OPENBLAS, 1);
  LOG(INFO) << "FastAlpaca SGEMV:";
  benchmarkGEMM(256, GEMV_LLMRT, 400);
  benchmarkGEMM(512, GEMV_LLMRT, 200);
  benchmarkGEMM(1024, GEMV_LLMRT, 50);
  benchmarkGEMM(2048, GEMV_LLMRT, 5);
  benchmarkGEMM(4096, GEMV_LLMRT, 1);
}
