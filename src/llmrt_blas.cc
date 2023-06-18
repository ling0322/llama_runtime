#include "llmrt_blas.h"

#include <stdlib.h>
#include "environment.h"
#include "blas_kernel.h"
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

typedef SGEMVImpl<SAXPYAvx2Kernel, SDOTAvx2Kernel> SGEMVImplAvx512;
typedef SGEMVImpl<SAXPYAvx2Kernel, SDOTAvx2Kernel> SGEMVImplAvx2;
typedef SGEMVImpl<SAXPYAvx2Kernel, SDOTAvx2Kernel> SGEMVImplDefault;

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

// -- class BatchSGEMV ----------

class BatchSGEMV {
 public:
  virtual ~BatchSGEMV() = default;
  virtual void apply(util::Span<const GEMVArgs> batchArgs) const = 0; 
};

template<class TSGEMVImpl>
class BatchSGEMVImpl : public BatchSGEMV {
 public:
  void apply(util::Span<const GEMVArgs> batchArgs) const override; 
};

template<class TSGEMVImpl>
void BatchSGEMVImpl<TSGEMVImpl>::apply(util::Span<const GEMVArgs> batchArgs) const {
  TSGEMVImpl sgemvImpl;
  for (const GEMVArgs &segmvArg: batchArgs) {
    sgemvImpl.apply(segmvArg);
  }
}


// -- class SGEMM ----------

typedef GEMMCommon<288, 512, 4096, SGEMM6x16DefaultKernel> SGEMMKernelDefault;
typedef GEMMCommon<288, 512, 4096, SGEMM6x16Avx2Kernel> SGEMMKernelAvx2;
typedef GEMMCommon<576, 512, 4096, SGEMM6x16Avx512Kernel> SGEMMKernelAvx512;

class SGEMM {
 public:
  virtual ~SGEMM() = default;
  virtual void apply(const GEMMArgs &args) const = 0;
};

template<class TGEMMKernel, class TGEMVImpl>
class SGEMMImpl : public SGEMM {
 public:
  void apply(const GEMMArgs &args) const override {
    if (args.M == 1) {
      rvmSgemv(args);
    } else if (args.N == 1) {
      mcvSgemv(args);
    } else {
      TGEMMKernel().Apply(
          args.TransA, args.TransB, args.M, args.N, args.K, args.A, args.lda, args.B, args.ldb,
          args.C, args.ldc);
    }
  }

 private:
  TGEMVImpl _sgemvImpl;

  // copy vector x to y.
  void scopy(int n, const float *x, int incx, float *y, int incy) const;

  // allocate n single float and returns the holder. the memory is 32 byte aligned.
  util::AutoCPtr<float> salloc(int64_t n) const;

  // row vector and matrix multiplication using SGEMV.
  void rvmSgemv(const GEMMArgs &args) const;

  // row vector and matrix multiplication using SGEMV.
  void mcvSgemv(const GEMMArgs &args) const;
};

typedef SGEMMImpl<SGEMMKernelAvx512, SGEMVImplAvx512> SGEMMImplAvx512;
typedef SGEMMImpl<SGEMMKernelAvx2, SGEMVImplAvx2> SGEMMImplAvx2;
typedef SGEMMImpl<SGEMMKernelDefault, SGEMVImplDefault> SGEMMImplDefault;

template<class TGEMMKernel, class TGEMVImpl>
void SGEMMImpl<TGEMMKernel, TGEMVImpl>::scopy(
    int n, const float *x, int incx, float *y, int incy) const {
  for (int i = 0; i < n; ++i) {
    y[i * incy] = x[i * incx];
  }
}

template<class TGEMMKernel, class TGEMVImpl>
util::AutoCPtr<float> SGEMMImpl<TGEMMKernel, TGEMVImpl>::salloc(int64_t n) const {
  return util::AutoCPtr<float>(
      reinterpret_cast<float *>(util::alloc32ByteAlignedMem(sizeof(float) * n)),
      util::free32ByteAlignedMem);
}

template<class TGEMMKernel, class TGEMVImpl>
void SGEMMImpl<TGEMMKernel, TGEMVImpl>::rvmSgemv(const GEMMArgs &args) const {
  CHECK(args.M == 1);

  util::AutoCPtr<float> packedA;
  bool needPackA = args.TransA && args.lda != 1;
  if (needPackA) {
    packedA = salloc(args.K);
    scopy(args.K, args.A, args.lda, packedA.get(), 1);
  }

  // fill C with zero.
  std::fill(args.C, args.C + args.N, 0.0f);

  GEMVArgs sgemvArgs;
  sgemvArgs.A = args.B;
  sgemvArgs.lda = args.ldb;
  sgemvArgs.x = needPackA ? packedA.get() : args.A;
  sgemvArgs.y = args.C;
  if (args.TransB) {
    sgemvArgs.M = args.N;
    sgemvArgs.N = args.K;
    sgemvArgs.TransA = false;
  } else {
    sgemvArgs.M = args.K;
    sgemvArgs.N = args.N;
    sgemvArgs.TransA = true;
  }

  _sgemvImpl.apply(sgemvArgs);
}

template<class TGEMMKernel, class TGEMVImpl>
void SGEMMImpl<TGEMMKernel, TGEMVImpl>::mcvSgemv(const GEMMArgs &args) const {
  CHECK(args.N == 1);

  util::AutoCPtr<float> packedB;
  util::AutoCPtr<float> packedC;

  bool needPackB = (!args.TransB) && args.ldb != 1;
  if (needPackB) {
    packedB = salloc(args.K);
    scopy(args.K, args.B, args.ldb, packedB.get(), 1);
  }

  bool needPackC = args.ldc != 1;
  if (needPackC) {
    packedC = salloc(args.M);
    std::fill(packedC.get(), packedC.get() + args.M, 0.0f);
  } else {
    std::fill(args.C, args.C + args.M, 0.0f);
  }

  GEMVArgs sgemvArgs;
  sgemvArgs.A = args.A;
  sgemvArgs.lda = args.lda;
  sgemvArgs.x = needPackB ? packedB.get() : args.B;
  sgemvArgs.y = args.C;
  if (args.TransA) {
    sgemvArgs.M = args.K;
    sgemvArgs.N = args.M;
    sgemvArgs.TransA = true;
  } else {
    sgemvArgs.M = args.M;
    sgemvArgs.N = args.K;
    sgemvArgs.TransA = false;
  }

  _sgemvImpl.apply(sgemvArgs);

  if (needPackC) {
    scopy(args.M, packedC.get(), 1, args.C, args.ldc);
  }
}

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
  for (const GEMMArgs &segmmArg: batchArgs) {
    sgemmImpl.apply(segmmArg);
  }
}

// -- class LLmRTBlas ----------

LLmRTBlasBackend LLmRTBlas::findBestBackend() {
  if (util::isAvx512Available()) {
    LOG(INFO) << "LLmRT GEMM: Use Avx512 backend.";
    return LLmRTBlasBackend::AVX512;
  } else if (util::isAvx2Available()) {
    LOG(INFO) << "LLmRT GEMM: Use Avx2 backend.";
    return LLmRTBlasBackend::AVX2;
  } else {
    LOG(WARN) << "LLmRT GEMM: fallback to default backend.";
    return LLmRTBlasBackend::AVX2;
  }
}

LLmRTBlas::LLmRTBlas() {
  initBackend();
}

LLmRTBlas::~LLmRTBlas() {}

void LLmRTBlas::initBackend() {
  switch (Environment::getLLmRTBlasBackend()) {
    case LLmRTBlasBackend::AVX512:
      _sgemmImpl = std::make_unique<SGEMMImplAvx512>();
      _sgemmBatchImpl = std::make_unique<BatchSGEMMImpl<SGEMMImplAvx512>>();
      _sgemvImpl = std::make_unique<SGEMVImplAvx512>();
      _sgemvBatchImpl = std::make_unique<BatchSGEMVImpl<SGEMVImplAvx512>>();
      _saxpyImpl = std::make_unique<SAXPYImpl<SAXPYAvx2Kernel>>();
      _sdotImpl = std::make_unique<SDOTImpl<SDOTAvx2Kernel>>();
      break;
    case LLmRTBlasBackend::AVX2:
      _sgemmImpl = std::make_unique<SGEMMImplAvx2>();
      _sgemmBatchImpl = std::make_unique<BatchSGEMMImpl<SGEMMImplAvx2>>();
      _sgemvImpl = std::make_unique<SGEMVImpl<SAXPYAvx2Kernel, SDOTAvx2Kernel>>();
      _sgemvBatchImpl = std::make_unique<BatchSGEMVImpl<SGEMVImplAvx2>>();
      _saxpyImpl = std::make_unique<SAXPYImpl<SAXPYAvx2Kernel>>();
      _sdotImpl = std::make_unique<SDOTImpl<SDOTAvx2Kernel>>();
      break;
    case LLmRTBlasBackend::DEFAULT:
      _sgemmImpl = std::make_unique<SGEMMImplDefault>();
      _sgemmBatchImpl = std::make_unique<BatchSGEMMImpl<SGEMMImplDefault>>();
      _sgemvImpl = std::make_unique<SGEMVImplDefault>();
      _sgemvBatchImpl = std::make_unique<BatchSGEMVImpl<SGEMVImplDefault>>();
      _saxpyImpl = std::make_unique<SAXPYImpl<SAXPYAvx2Kernel>>();
      _sdotImpl = std::make_unique<SDOTImpl<SDOTAvx2Kernel>>();
      break;
    default:
      NOT_IMPL();
  }
}

void LLmRTBlas::sgemm(const GEMMArgs &args) const {
  return _sgemmImpl->apply(args);
}

void LLmRTBlas::sgemv(const GEMVArgs &args) const {
  return _sgemvImpl->apply(args);
}

void LLmRTBlas::sgemmBatch(util::Span<const GEMMArgs> batchArgs) const {
  return _sgemmBatchImpl->apply(batchArgs);
}

void LLmRTBlas::sgemvBatch(util::Span<const GEMVArgs> batchArgs) const {
  return _sgemvBatchImpl->apply(batchArgs);
}

}  // namespace nn
}  // namespace llama
