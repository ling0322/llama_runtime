// A default cpu-based GEMM
#pragma once

#include <stdint.h>
#include <memory>
#include "pmpack/pmpack.h"
#include "pmpack/gemm_common.h"
#include "pmpack/gemm_kernel.h"
#include "util/util.h"

namespace llama {
namespace nn {

class SGEMV {
 public:
  virtual ~SGEMV() = default;
  virtual void apply(
      bool transA,
      int M,
      int N,
      const float *A,
      int lda,
      const float *x,
      float *y) const = 0; 
};

template<class TSAxpyKernel, class TSDotKernel>
class SGEMVImpl : public SGEMV {
 public:
  void apply(
      bool transA,
      int M,
      int N,
      const float *A,
      int lda,
      const float *x,
      float *y) const override;

 private:
  void applyTransA(
      int M,
      int N,
      const float *A,
      int lda,
      const float *x,
      float *y) const;
  void applyNoTransA(
      int M,
      int N,
      const float *A,
      int lda,
      const float *x,
      float *y) const;
};

typedef SGEMVImpl<SAXPYAvx2Kernel, SDOTAvx2Kernel> SGEMVImplAvx512;
typedef SGEMVImpl<SAXPYAvx2Kernel, SDOTAvx2Kernel> SGEMVImplAvx2;
typedef SGEMVImpl<SAXPYAvx2Kernel, SDOTAvx2Kernel> SGEMVImplDefault;

template<class TSAxpyKernel, class TSDotKernel>
void SGEMVImpl<TSAxpyKernel, TSDotKernel>::applyTransA(
    int M,
    int N,
    const float *A,
    int lda,
    const float *x,
    float *y) const {
  // dimemsion of (x, y) is (M, N)
  const float *pa = A;
  for (int m = 0; m < M; ++m) {
    TSAxpyKernel::callKernel(N, x[m], pa, y);
    pa += lda;
  }
}

template<class TSAxpyKernel, class TSDotKernel>
void SGEMVImpl<TSAxpyKernel, TSDotKernel>::applyNoTransA(
    int M,
    int N,
    const float *A,
    int lda,
    const float *x,
    float *y) const {
  // dimemsion of (x, y) is (N, M)
  const float *pa = A;
  for (int m = 0; m < M; ++m) {
    y[m] += TSDotKernel::callKernel(N, pa, x);
    pa += lda;
  }
}

template<class TSAxpyKernel, class TSDotKernel>
void SGEMVImpl<TSAxpyKernel, TSDotKernel>::apply(
    bool transA,
    int M,
    int N,
    const float *A,
    int lda,
    const float *x,
    float *y) const {
  if (transA) {
    applyTransA(M, N, A, lda, x, y);
  } else {
    applyNoTransA(M, N, A, lda, x, y);
  }
}

}  // namespace nn
}  // namespace llama
