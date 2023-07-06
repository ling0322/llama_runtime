// A default cpu-based GEMM
#pragma once

#include <stdint.h>
#include <memory>
#include "common/environment.h"
#include "pmpack/pmpack.h"
#include "pmpack/gemm_common.h"
#include "pmpack/gemm_kernel.h"
#include "pmpack/sgemv.h"
#include "util/util.h"

namespace llama {
namespace nn {


// -- class SGEMM ----------

typedef GEMMCommon<288, 512, 4096, SGEMM6x16DefaultKernel> SGEMMKernelDefault;
typedef GEMMCommon<288, 512, 4096, SGEMM6x16Avx2Kernel> SGEMMKernelAvx2;
typedef GEMMCommon<576, 512, 4096, SGEMM12x32Avx512Kernel> SGEMMKernelAvx512;

class SGEMM {
 public:
  virtual ~SGEMM() = default;

  virtual void apply(
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const float *A,
      int lda,
      const float *B,
      int ldb,
      float *C,
      int ldc) const = 0;

  virtual void applyBatch(
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

template<class TGEMMKernel, class TGEMVImpl>
class SGEMMImpl : public SGEMM {
 public:
  void apply(
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const float *A,
      int lda,
      const float *B,
      int ldb,
      float *C,
      int ldc) const override {
    if (M == 1) {
      applyRowVectorA(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
    } else if (N == 1) {
      applyColumnVectorB(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
    } else {
      TGEMMKernel().Apply(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
    }
  }

  void applyBatch(
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

 private:
  TGEMVImpl _sgemvImpl;

  // row vector and matrix multiplication using SGEMV.
  void applyRowVectorA(
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const float *A,
      int lda,
      const float *B,
      int ldb,
      float *C,
      int ldc) const;

  // row vector and matrix multiplication using SGEMV.
  void applyColumnVectorB(
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const float *A,
      int lda,
      const float *B,
      int ldb,
      float *C,
      int ldc) const;
};

typedef SGEMMImpl<SGEMMKernelAvx512, SGEMVImplAvx512> SGEMMImplAvx512;
typedef SGEMMImpl<SGEMMKernelAvx2, SGEMVImplAvx2> SGEMMImplAvx2;
typedef SGEMMImpl<SGEMMKernelDefault, SGEMVImplDefault> SGEMMImplDefault;

template<class TGEMMKernel, class TGEMVImpl>
void SGEMMImpl<TGEMMKernel, TGEMVImpl>::applyBatch(
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
  TGEMMKernel gemmKernel;
  for (int i = 0; i < batchSize; ++i) {
    const float *A = batchA[i];
    const float *B = batchB[i];
    float *C = batchC[i];
    if (M == 1) {
      applyRowVectorA(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
    } else if (N == 1) {
      applyColumnVectorB(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
    } else {
      gemmKernel.Apply(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
    }
  }
}

template<class TGEMMKernel, class TGEMVImpl>
void SGEMMImpl<TGEMMKernel, TGEMVImpl>::applyRowVectorA(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float *C,
    int ldc) const {
  CHECK(M == 1);

  util::AutoCPtr<float> packedA;
  bool needPackA = transA && lda != 1;
  if (needPackA) {
    packedA = salloc(K);
    scopy(K, A, lda, packedA.get(), 1);
  }

  // fill C with zero.
  std::fill(C, C + N, 0.0f);

  _sgemvImpl.apply(
    !transB,
    transB ? N : K,
    transB ? K : N,
    B,
    ldb,
    needPackA ? packedA.get() : A,
    C);
}

template<class TGEMMKernel, class TGEMVImpl>
void SGEMMImpl<TGEMMKernel, TGEMVImpl>::applyColumnVectorB(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float *C,
    int ldc) const {
  CHECK(N == 1);

  util::AutoCPtr<float> packedB;
  util::AutoCPtr<float> packedC;

  bool needPackB = (!transB) && ldb != 1;
  if (needPackB) {
    packedB = salloc(K);
    scopy(K, B, ldb, packedB.get(), 1);
  }

  bool needPackC = ldc != 1;
  if (needPackC) {
    packedC = salloc(M);
    std::fill(packedC.get(), packedC.get() + M, 0.0f);
  } else {
    std::fill(C, C + M, 0.0f);
  }

  _sgemvImpl.apply(
      transA,
      transA ? K : M,
      transA ? M : K,
      A,
      lda,
      needPackB ? packedB.get() : B,
      needPackB ? packedC.get() : C);

  if (needPackC) {
    scopy(M, packedC.get(), 1, C, ldc);
  }
}

}  // namespace nn
}  // namespace llama
