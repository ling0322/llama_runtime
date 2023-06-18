// A default cpu-based GEMM
#pragma once

#include <stdint.h>
#include <memory>
#include "util.h"

namespace llama {
namespace nn {

enum class LLmRTBlasBackend {
  DEFAULT,
  AVX2,
  AVX512
};

class SGEMM;
class BatchSGEMM;
class SGEMV;
class BatchSGEMV;
class SAXPY;
class SDOT;

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

// arguments for GEMV.
struct GEMVArgs {
  bool TransA;
  int M;
  int N;
  const float *A;
  int lda;
  const float *x;
  float *y;

  GEMVArgs();
};

// interface for matrix multiplication.
class LLmRTBlas {
 public:
  // find the best backend for GEMM. By design, this function would be called in 
  static LLmRTBlasBackend findBestBackend();

  LLmRTBlas();
  ~LLmRTBlas();

  // matrix-matrix multiplication. 
  void sgemm(const GEMMArgs &args) const;

  // batched matrix-matrix multiplication.
  void sgemmBatch(util::Span<const GEMMArgs> batchArgs) const;

  // matrix-vector multiplication. 
  void sgemv(const GEMVArgs &args) const;

  // matrix-vector multiplication. 
  void sgemvBatch(util::Span<const GEMVArgs> batchArgs) const;

  // y += a * x
  void saxpy(int64_t n, float a, float *x, float *y);

  // return x dot y
  float sdot(int64_t n, float *x, float *y);

 private:
  std::unique_ptr<SGEMM> _sgemmImpl;
  std::unique_ptr<BatchSGEMM> _sgemmBatchImpl;
  std::unique_ptr<SGEMV> _sgemvImpl;
  std::unique_ptr<BatchSGEMV> _sgemvBatchImpl;
  std::unique_ptr<SAXPY> _saxpyImpl;
  std::unique_ptr<SDOT> _sdotImpl;

  // choose the backend for each operations.
  void initBackend();
};


}  // namespace nn
}  // namespace llama
