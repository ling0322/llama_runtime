#include "gemm.h"

#include <stdlib.h>
#include "gemm_common.h"
#include "gemm_kernel.h"
#include "log.h"
#include "util.h"

namespace llama {
namespace nn {

// -- class GEMMArgs ----------

GEMMArgs::GEMMArgs()
    : TransA(false),
      TransB(false),
      M(0),
      N(0),
      K(0),
      A(nullptr),
      lda(0),
      B(nullptr),
      ldb(0),
      C(nullptr),
      ldc(0) {}


// -- class GEMVArgs ----------

GEMVArgs::GEMVArgs()
    : TransA(false),
      M(0),
      N(0),
      A(nullptr),
      lda(0),
      x(nullptr),
      y(nullptr) {}

// -- class SGEMM ----------

typedef GEMMCommon<288, 512, 4096, SGEMM6x16DefaultKernel> SGEMMFallback;
typedef GEMMCommon<288, 512, 4096, SGEMM6x16Avx2Kernel> SGEMMAvx2;
typedef GEMMCommon<576, 512, 4096, SGEMM6x16Avx512Kernel> SGEMMAvx512;

class SGEMM {
 public:
  virtual ~SGEMM() = default;
  virtual void apply(const GEMMArgs &args) const = 0;
};

template<class TImpl>
class SGEMMImpl : public SGEMM {
 public:
  void apply(const GEMMArgs &args) const override {
    TImpl().Apply(
        args.TransA, args.TransB, args.M, args.N, args.K, args.A, args.lda, args.B, args.ldb,
        args.C, args.ldc);
  }

 private:
  TImpl _sgemmImpl;
};

// -- class BatchSGEMM ----------

class BatchSGEMM {
 public:
  virtual ~BatchSGEMM() = default;
  virtual void apply(util::Span<const GEMMArgs> batchArgs) const = 0;
};

template<class TSGEMMImpl>
class BatchSGEMMImpl : public BatchSGEMM {
 public:  
  void apply(util::Span<const GEMMArgs> batchArgs) const override;
};

template<class TSGEMMImpl>
void BatchSGEMMImpl<TSGEMMImpl>::apply(util::Span<const GEMMArgs> batchArgs) const {
  TSGEMMImpl sgemmImpl;
  for (int batch = 0; batch < batchSize; ++batch) {

  }
}

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
};

// -- class GEMV ----------

class SGEMV {
 public:
  virtual ~SGEMV() = default;
  virtual void apply(const GEMVArgs &args) const = 0; 
};

template<class TSAxpyKernel, class TSDotKernel>
class SGEMVImpl : public SGEMV {
 public:
  void apply(const GEMVArgs &args) const override;

 private:
  void applyTransA(const GEMVArgs &args) const;
  void applyNoTransA(const GEMVArgs &args) const;
};

template<class TSAxpyKernel, class TSDotKernel>
void SGEMVImpl<TSAxpyKernel, TSDotKernel>::applyTransA(const GEMVArgs &args) const {
  // dimemsion of (x, y) is (M, N)
  const float *pa = args.A;
  for (int m = 0; m < args.M; ++m) {
    TSAxpyKernel::callKernel(args.N, args.x[m], pa, args.y);
    pa += args.lda;
  }
}

template<class TSAxpyKernel, class TSDotKernel>
void SGEMVImpl<TSAxpyKernel, TSDotKernel>::applyNoTransA(const GEMVArgs &args) const {
  // dimemsion of (x, y) is (N, M)
  const float *pa = args.A;
  for (int m = 0; m < args.M; ++m) {
    args.y[m] += TSDotKernel::callKernel(args.N, pa, args.x);
    pa += args.lda;
  }
}

template<class TSAxpyKernel, class TSDotKernel>
void SGEMVImpl<TSAxpyKernel, TSDotKernel>::apply(const GEMVArgs &args) const {
  if (args.TransA) {
    applyTransA(args);
  } else {
    applyNoTransA(args);
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

void GEMM::sgemm(const GEMMArgs &args) const {
  return _sgemmImpl->apply(args);
}

void GEMM::sgemv(const GEMVArgs &args) const {
  return _sgemvImpl->apply(args);
}

}  // namespace nn
}  // namespace llama
