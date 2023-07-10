// A default cpu-based GEMM
#pragma once

#include <stdint.h>
#include <memory>
#include "pmpack/pmpack.h"
#include "pmpack/util.h"
#include "pmpack/gemm_common.h"
#include "pmpack/gemm_kernel.h"
#include "pmpack/sgemm.h"

// IGemmFp32QInt4Fp32 -> GemmFp32QInt4Fp32Impl -> GemmFp32QInt4Fp32Kernel

namespace pmpack {

class IGemmFp32QInt4Fp32 {
 public:
  virtual ~IGemmFp32QInt4Fp32() = default;
  virtual void apply(
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const float *A,
      int lda,
      const void *B,
      const float *scaleDataB,
      int groupSizeB,
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
      const void *const *batchB,
      const float *const *batchScaleB,
      int groupSizeB,
      float *const *batchC,
      int ldc) const = 0;
};

template<class TQGemmKernel, class TQDotKernel>
class GemmFp32QInt4Fp32Impl : public IGemmFp32QInt4Fp32 {
 public:
  void apply(
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const float *A,
      int lda,
      const void *B,
      const float *scaleDataB,
      int groupSizeB,
      float *C,
      int ldc) const override {
    if (M == 1) {
      appleRowVectorA(transA, transB, M, N, K, A, lda, B, scaleDataB, groupSizeB, C, ldc);
    } else if (N == 1) {
      NOT_IMPL();
    } else {
      TQGemmKernel().apply(transA, transB, M, N, K, A, lda, B, scaleDataB, groupSizeB, C, ldc);
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
      const void *const *batchB,
      const float *const *batchScaleB,
      int groupSizeB,
      float *const *batchC,
      int ldc) const override;

 private:
  void appleRowVectorA(
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const float *A,
      int lda,
      const void *B,
      const float *scaleDataB,
      int groupSizeB,
      float *C,
      int ldc) const;
};

template<class TGEMMKernel>
class GemmFp32QInt4Fp32Kernel {
 public:
  void apply(
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const float *A,
      int lda,
      const void *B,
      const float *scaleDataB,
      int groupSizeB,
      float *C,
      int ldc);

 private:
  ly::c_ptr<float> _dequantData;
  int64_t _dequantNumEl;
};

template<class TGEMMKernel>
void GemmFp32QInt4Fp32Kernel<TGEMMKernel>::apply(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const void *B,
    const float *scaleDataB,
    int groupSizeB,
    float *C,
    int ldc) {
  CHECK(transB);
  CHECK(groupSizeB == K);

  if (_dequantData.get() == nullptr) {
    _dequantData = salloc(M * N);
    _dequantNumEl = M * N;
  } else {
    CHECK(M * N == _dequantNumEl);
  }

  QInt4Block qblock(
      const_cast<void *>(B),
      const_cast<float *>(scaleDataB),
      groupSizeB,
      K,
      N,
      true);
  Block blockB{_dequantData.get(), N, K, N, true};

  qblock.dequantizeTo(blockB);
  TGEMMKernel().Apply(
      transA,
      transB,
      M,
      N,
      K,
      A,
      lda,
      blockB.data,
      blockB.stride,
      C,
      ldc);
}

typedef GemmFp32QInt4Fp32Impl<GemmFp32QInt4Fp32Kernel<SGEMMKernelAvx512>,
                              DOTFp32Int4Fp32Avx2Kernel> GemmFp32QInt4Fp32Avx512;
typedef GemmFp32QInt4Fp32Impl<GemmFp32QInt4Fp32Kernel<SGEMMKernelAvx2>,
                              DOTFp32Int4Fp32Avx2Kernel> GemmFp32QInt4Fp32Avx2;
typedef GemmFp32QInt4Fp32Impl<GemmFp32QInt4Fp32Kernel<SGEMMKernelDefault>,
                              DOTFp32Int4Fp32FallbackKernel> GemmFp32QInt4Fp32Fallback;

template<class TQGemmKernel, class TQDotKernel>
void GemmFp32QInt4Fp32Impl<TQGemmKernel, TQDotKernel>::applyBatch(
    int batchSize,
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *const *batchA,
    int lda,
    const void *const *batchB,
    const float *const *batchScaleB,
    int groupSizeB,
    float *const *batchC,
    int ldc) const {
  TQGemmKernel qgemmKernel;

  for (int i = 0; i < batchSize; ++i) {
    const float *A = batchA[i];
    const void *B = batchB[i];
    const float *scaleB = batchScaleB[i];
    float *C = batchC[i];

    if (M == 1) {
      appleRowVectorA(transA, transB, M, N, K, A, lda, B, scaleB, groupSizeB, C, ldc);
    } else if (N == 1) {
      NOT_IMPL();
    } else {
      qgemmKernel.apply(transA, transB, M, N, K, A, lda, B, scaleB, groupSizeB, C, ldc);
    }
  }
}

template<class TQGemmKernel, class TQDotKernel>
void GemmFp32QInt4Fp32Impl<TQGemmKernel, TQDotKernel>::appleRowVectorA(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const void *B,
    const float *scaleDataB,
    int groupSizeB,
    float *C,
    int ldc) const {
  CHECK(transB);
  CHECK(M == 1);
  CHECK(K % 32 == 0);

  bool needPackA = transA && lda != 1;
  if (needPackA) {
    NOT_IMPL();
  }

  // GEMV
  const uint8_t *Bi = reinterpret_cast<const uint8_t*>(B);
  for (int i = 0; i < N; ++i) {
    C[i] = TQDotKernel::apply(K, A, Bi, scaleDataB[i]);
    Bi += K / 2;
  }
}

}  // namespace pmpack
