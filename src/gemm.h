// A default cpu-based GEMM

#ifndef GLM_RUNTIME_GLAS_H_
#define GLM_RUNTIME_GLAS_H_

#include <stdint.h>
#include <memory>
#include "log.h"

namespace llama {
namespace nn {

enum class GEMMBackend {
  DEFAULT,
  AVX2,
  AVX512
};

class SGEMM;
class SGEMV;
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
class GEMM {
 public:
  GEMM();
  ~GEMM();

  // matrix-matrix multiplication. 
  void sgemm(const GEMMArgs &args) const;

  // matrix-vector multiplication. 
  void sgemv(const GEMVArgs &args) const;

  // y += a * x
  void saxpy(int64_t n, float a, float *x, float *y);

  // return x dot y
  float sdot(int64_t n, float *x, float *y);

 private:
  std::unique_ptr<SGEMM> _sgemmImpl;
  std::unique_ptr<SGEMV> _sgemvImpl;
  std::unique_ptr<SAXPY> _saxpyImpl;
  std::unique_ptr<SDOT> _sdotImpl;

  // choose the backend for each operations.
  void chooseBackend();
};


}  // namespace nn
}  // namespace llama

#endif  // GLM_RUNTIME_GLAS_H_
