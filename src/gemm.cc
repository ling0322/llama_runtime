#include "gemm.h"

#include <stdlib.h>
#include "gemm_common.h"
#include "gemm_kernel.h"
#include "log.h"
#include "util.h"

namespace llama {
namespace nn {

// -- class SGEMM ----------

typedef GEMMCommon<288, 512, 4096, SGEMM6x16DefaultKernel> SGEMMFallback;
typedef GEMMCommon<288, 512, 4096, SGEMM6x16Avx2Kernel> SGEMMAvx2;
typedef GEMMCommon<576, 512, 4096, SGEMM6x16Avx512Kernel> SGEMMAvx512;

class SGEMM {
 public:
  virtual ~SGEMM() = default;
  virtual void apply(
      bool TransA, bool TransB, int M, int N, int K, const float *A, int lda,
      const float *B, int ldb, float *C, int ldc) const = 0;
};

template<class TImpl>
class SGEMMImpl : public SGEMM {
 public:
  void apply(
      bool TransA, bool TransB, int M, int N, int K, const float *A, int lda,
      const float *B, int ldb, float *C, int ldc) const override {
    TImpl().Apply(TransA, TransB, M, N, K, A, lda, B, ldb, C, ldc);
  }

 private:
  TImpl _sgemmImpl;
};

// -- class SAXPY ----------

class SAXPY {
 public:
  virtual ~SAXPY() = default;
  virtual void apply(int64_t n, float a, const float *x, float *y) const = 0;
};

template<class TImpl>
class SAXPYImpl : public SAXPY {
 public:
  void apply(int64_t n, float a, const float *x, float *y) const override {
    TImpl::callKernel(n, a, x, y);
  }

 private:
  TImpl _saxpyImpl;
};

// -- class SDOT ----------

class SDOT {
 public:
  virtual ~SDOT() = default;
  virtual float apply(int64_t n, const float *x, const float *y) const = 0;
};

template<class TImpl>
class SDOTImpl : public SDOT {
 public:
  float apply(int64_t n, const float *x, const float *y) const override {
    return TImpl::callKernel(n, x, y);
  }

 private:
  TImpl _sdotImpl;
};

// -- class GEMV ----------

class SGEMV {
 public:
  virtual ~SGEMV() = default;
  virtual void apply(
      bool TransA, int M, int N, const float *A, int lda, const float *x, float *y) const = 0; 
};

template<class TSAxpyKernel, class TSDotKernel>
class SGEMVImpl : public SGEMV {
 public:
  void apply(
      bool TransA, int M, int N, const float *A, int lda, const float *x, float *y) const override;

 private:
  void applyTransA(int M, int N, const float *A, int lda, const float *x, float *y) const;
  void applyNoTransA(int M, int N, const float *A, int lda, const float *x, float *y) const;
};

template<class TSAxpyKernel, class TSDotKernel>
void SGEMVImpl<TSAxpyKernel, TSDotKernel>::applyTransA(
    int M, int N, const float *A, int lda, const float *x, float *y) const {
  // dimemsion of (x, y) is (M, N)
  const float *pa = A;
  for (int m = 0; m < M; ++m) {
    TSAxpyKernel::callKernel(N, x[m], pa, y);
    pa += lda;
  }
}

template<class TSAxpyKernel, class TSDotKernel>
void SGEMVImpl<TSAxpyKernel, TSDotKernel>::applyNoTransA(
    int M, int N, const float *A, int lda, const float *x, float *y) const {
  // dimemsion of (x, y) is (N, M)
  const float *pa = A;
  for (int m = 0; m < M; ++m) {
    y[m] += TSDotKernel::callKernel(N, pa, x);
    pa += lda;
  }
}

template<class TSAxpyKernel, class TSDotKernel>
void SGEMVImpl<TSAxpyKernel, TSDotKernel>::apply(
    bool TransA, int M, int N, const float *A, int lda, const float *x, float *y) const {
  if (TransA) {
    applyTransA(M, N, A, lda, x, y);
  } else {
    applyNoTransA(M, N, A, lda, x, y);
  }
}

// -- class GEMM ----------

GEMM::GEMM() {
  chooseBackend();
}

GEMM::~GEMM() {}

void GEMM::chooseBackend() {
  if (util::isAvx512Available()) {
    _sgemmImpl = std::make_unique<SGEMMImpl<SGEMMAvx512>>();
    _sgemvImpl = std::make_unique<SGEMVImpl<SAXPYAvx2Kernel, SDOTAvx2Kernel>>();
    _saxpyImpl = std::make_unique<SAXPYImpl<SAXPYAvx2Kernel>>();
    _sdotImpl = std::make_unique<SDOTImpl<SDOTAvx2Kernel>>();
  } else if (util::isAvx2Available()) {
    _sgemmImpl = std::make_unique<SGEMMImpl<SGEMMAvx2>>();
    _sgemvImpl = std::make_unique<SGEMVImpl<SAXPYAvx2Kernel, SDOTAvx2Kernel>>();
    _saxpyImpl = std::make_unique<SAXPYImpl<SAXPYAvx2Kernel>>();
    _sdotImpl = std::make_unique<SDOTImpl<SDOTAvx2Kernel>>();
  } else {
    _sgemmImpl = std::make_unique<SGEMMImpl<SGEMMFallback>>();
    _sgemvImpl = std::make_unique<SGEMVImpl<SAXPYAvx2Kernel, SDOTAvx2Kernel>>();
    _saxpyImpl = std::make_unique<SAXPYImpl<SAXPYAvx2Kernel>>();
    _sdotImpl = std::make_unique<SDOTImpl<SDOTAvx2Kernel>>();
  }
}

void GEMM::sgemm(
    bool TransA, bool TransB, int M, int N, int K, const float *A, int lda,
    const float *B, int ldb, float *C, int ldc) {
  return _sgemmImpl->apply(TransA, TransB, M, N, K, A, lda, B, ldb, C, ldc);
}

void GEMM::sgemv(
    bool TransA, int M, int N, const float *A, int lda, const float *x, float *y) const {
  return _sgemvImpl->apply(TransA, M, N, A, lda, x, y);
}

}  // namespace nn
}  // namespace llama
