// A default cpu-based GEMM

#ifndef GLM_RUNTIME_GLAS_H_
#define GLM_RUNTIME_GLAS_H_

#include <stdint.h>
#include <memory>
#include "log.h"

namespace llama {
namespace nn {

enum class GEMMBackend {
  kDefault,
  kAvx2,
  kAvx512
};

// interface for matrix multiplication.
class GEMM {
 public:
  GEMM();

  // float32 matrix multiplication. 
  void sgemm(bool TransA, bool TransB,
      int M, int N, int K,
      const float *A, int lda,
      const float *B, int ldb,
      float *C, int ldc);

 private:
  GEMMBackend sgemm_backend_;
  
};

}  // namespace nn
}  // namespace llama

#endif  // GLM_RUNTIME_GLAS_H_
