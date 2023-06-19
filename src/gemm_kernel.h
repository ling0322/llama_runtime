// A default cpu-based GEMM
#pragma once

#include <stdint.h>
#include <memory>
#include "util.h"

namespace llama {
namespace nn {

enum class CPUMathBackend {
  DEFAULT,
  AVX2,
  AVX512
};

// get the best backend for GEMM according to CPU and OS.
CPUMathBackend findBestCpuMathBackend();

// arguments for GEMM.
struct GEMMArgs {
  bool TransA;
  bool TransB;
  int M;
  int N;
  int K;
  const float *A;
  int lda;
  const float *B;
  int ldb;
  float *C;
  int ldc;

  GEMMArgs();
};

class SGEMM {
 public:
  static std::unique_ptr<SGEMM> create();

  virtual ~SGEMM() = default;
  virtual void apply(const GEMMArgs &args) const = 0;
};

class BatchSGEMM {
 public:
  static std::unique_ptr<BatchSGEMM> create();

  virtual ~BatchSGEMM() = default;
  virtual void apply(util::Span<const GEMMArgs> batchArgs) const = 0;
};

}  // namespace nn
}  // namespace llama
