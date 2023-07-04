// A default cpu-based GEMM
#pragma once

#include <stdint.h>
#include <memory>
#include "util/util.h"
#include "pmpack/pmpack.h"

namespace llama {
namespace nn {

// -- class BatchSGEMM ----------

class BatchSGEMM {
 public:
  static std::unique_ptr<BatchSGEMM> create();

  virtual ~BatchSGEMM() = default;
  virtual void apply(
      int batchSize,
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const float *const *batchA,
      int lda,
      const float *const *batchB,
      int ldb,
      float *const *batchC,
      int ldc) const = 0;
};

template<class TSGEMMImpl>
class BatchSGEMMImpl : public BatchSGEMM {
 public:  
  void apply(
      int batchSize,
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const float *const *batchA,
      int lda,
      const float *const *batchB,
      int ldb,
      float *const *batchC,
      int ldc) const override;
};

template<class TSGEMMImpl>
void BatchSGEMMImpl<TSGEMMImpl>::apply(
    int batchSize,
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *const *batchA,
    int lda,
    const float *const *batchB,
    int ldb,
    float *const *batchC,
    int ldc) const {
  TSGEMMImpl sgemmImpl;
  for (int i = 0; i < batchSize; ++i) {
    const float *A = batchA[i];
    const float *B = batchB[i];
    float *C = batchC[i];
    sgemmImpl.apply(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
  }
}

}  // namespace nn
}  // namespace llama
