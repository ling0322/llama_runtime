#include "gemm_kernel.h"

#include <stdlib.h>
#include "environment.h"
#include "log.h"
#include "util.h"

namespace llama {
namespace nn {

// -- function findBestCpuMathBackend ----------

CPUMathBackend findBestCpuMathBackend() {
  if (util::isAvx512Available()) {
    LOG(INFO) << "LLmRT GEMM: Use Avx512 backend.";
    return CPUMathBackend::AVX512;
  } else if (util::isAvx2Available()) {
    LOG(INFO) << "LLmRT GEMM: Use Avx2 backend.";
    return CPUMathBackend::AVX2;
  } else {
    LOG(WARN) << "LLmRT GEMM: fallback to default backend.";
    return CPUMathBackend::AVX2;
  }
}

// -- optimized micro-kernels ---------

void sgemmKernel12x32Avx512(int64_t kc, float *a, float *b, float *c, int64_t rs_c);
void sgemmKernel6x16Avx2(int64_t kc, float *a, float *b, float *c, int64_t rs_c);
void saxpyKernelAvx2(int64_t n, float a, const float *x, float *y);
float sdotKernelAvx2(int64_t n, const float *x, const float *y);

// -- fallback micro-kernels ---------

void sgemmKernel6x16Fallback(int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR
  constexpr int64_t MR = 6;
  constexpr int64_t NR = 16;

  for (int k = 0; k < kc; ++k) {
    float *Ak = a + k * MR;
    for (int m = 0; m < MR; ++m) {
      float *Cm = c + m * rs_c;
      float Akm = Ak[m];
      float *Bk = b + k * NR;
      
      for (int n = 0; n < NR; ++n) {
        Cm[n] += Akm * Bk[n];
      }
    }
  }
}

// -- micro-kernels ----------

class SGEMM6x16DefaultKernel {
 public:
  static constexpr int MR = 6;
  static constexpr int NR = 16;
  static inline void callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
    sgemmKernel6x16Fallback(kc, a, b, c, rs_c);
  }
};

class SGEMM6x16Avx2Kernel {
 public:
  static constexpr int MR = 6;
  static constexpr int NR = 16;
  static inline void callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
    sgemmKernel6x16Avx2(kc, a, b, c, rs_c);
  }
};

class SGEMM12x32Avx512Kernel {
 public:
  static constexpr int MR = 12;
  static constexpr int NR = 32;
  static inline void callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
    sgemmKernel12x32Avx512(kc, a, b, c, rs_c);
  }
};

class SAXPYAvx2Kernel {
 public:
  static inline void callKernel(int64_t n, float a, const float *x, float *y) {
    saxpyKernelAvx2(n, a, x, y);
  }
};

class SDOTAvx2Kernel {
 public:
  static inline float callKernel(int64_t n, const float *x, const float *y) {
    return sdotKernelAvx2(n, x, y);
  }
};

// -- common part for GEMM ----------

struct Block {
  float *data;
  int32_t stride;
  int32_t numRows;
  int32_t numCols;
  bool transposed;

  constexpr Block sliceRow(int row, int nr);
  constexpr Block sliceCol(int col, int nc);
  constexpr Block slice(int row, int col, int nr, int nc);
  constexpr void copyTo(Block tgt);
  constexpr Block T();
  constexpr void fillZero();

 private:
  constexpr void copy0(Block tgt);
  constexpr void copy1(Block tgt);
  constexpr void copy2(Block tgt);
  constexpr void copy3(Block tgt);
};

struct PackedBlock {
  float *data;
  int32_t packSize;
  int32_t numRows;
  int32_t numBlocks;

  constexpr Block block(int i);
};

template<int MC, int KC, int NC, class TKernel>
class GEMMCommon {
 public:
  GEMMCommon();
  ~GEMMCommon();

  static constexpr int MR = TKernel::MR;
  static constexpr int NR = TKernel::NR;

  // Compute C <- A * B
  void Apply(
      bool TransA, bool TransB,
      int M, int N, int K,
      const float *A, int lda,
      const float *B, int ldb,
      float *C, int ldc);

 private:
  float *_packedBuffer;

  Block _bufferA;
  Block _bufferB;

  Block _inputA;
  Block _inputB;
  Block _inputC;

  void split0ByNC();
  void split1ByKC(Block Bn, Block Cj);
  void split2ByMC(Block Ak, PackedBlock Bp, Block Cj);
  void split3ByNR(PackedBlock Ap, PackedBlock Bp, Block Cij);
  void split4ByMR(PackedBlock Ap, Block Bpr, Block Cijn);
  void callKernel(Block Amkr, Block Bknr, Block Cijmn);
};

constexpr Block Block::sliceRow(int row, int nr) {
  return slice(row, 0, nr, numCols);
}
constexpr Block Block::sliceCol(int col, int nc) {
  return slice(0, col, numRows, nc);
}
constexpr Block Block::slice(int row, int col, int nr, int nc) {
  return Block {
    data + (transposed ? row + col * stride : row * stride + col),
    stride,
    nr,
    nc,
    transposed
  };
}

// copy NoTrans -> NoTrans
constexpr void Block::copy0(Block tgt) {
  for (int r = 0; r < numRows; ++r) {
    int tgtOffset = r * tgt.stride;
    int srcOffset = r * stride;
    for (int c = 0; c < numCols; ++c) {
      tgt.data[tgtOffset + c] = data[srcOffset + c];
    }
  }
}

// copy Trans -> NoTrans
constexpr void Block::copy1(Block tgt) {
  for (int r = 0; r < numRows; ++r) {
    int tgtOffset = r * tgt.stride;
    for (int c = 0; c < numCols; ++c) {
      tgt.data[tgtOffset + c] = data[r + c * stride];
    }
  }
}

// copy NoTrans -> Trans
constexpr void Block::copy2(Block tgt) {
  for (int r = 0; r < numRows; ++r) {
    int srcOffset = r * stride;
    for (int c = 0; c < numCols; ++c) {
      tgt.data[r + c * tgt.stride] = data[srcOffset + c];
    }
  }
}

// copy Trans -> Trans
constexpr void Block::copy3(Block tgt) {
  for (int c = 0; c < numCols; ++c) {
    int srcOffset = c * stride;
    int tgtOffset = c * tgt.stride;
    for (int r = 0; r < numRows; ++r) {
        tgt.data[r + tgtOffset] = data[r + srcOffset];
    }
  }
}

constexpr void Block::copyTo(Block tgt) {
  ASSERT(numRows == tgt.numRows);
  ASSERT(numCols == tgt.numCols);

  if ((!transposed) && (!tgt.transposed)) {
    copy0(tgt);
  } else if (transposed && (!tgt.transposed)) {
    copy1(tgt);
  } else if ((!transposed) && tgt.transposed) {
    copy2(tgt);
  } else if (transposed && tgt.transposed) {
    copy3(tgt);
  }
}
constexpr Block Block::T() {
  return Block {
    data,
    stride,
    numCols,
    numRows,
    !transposed
  };
}
constexpr void Block::fillZero() {
  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      if (transposed) {
        data[r + c * stride] = 0.0f;
      } else {
        data[r * stride + c] = 0.0f;
      }
    }
  }
}
constexpr Block PackedBlock::block(int i) {
  return Block {
    data + packSize * numRows * i,
    packSize,
    numRows,
    packSize,
    false
  };
}

template<int MC, int KC, int NC, class TKernel>
inline GEMMCommon<MC, KC, NC, TKernel>::GEMMCommon() {
  int packedSize = (MC * KC + KC * NC) * sizeof(float);
  _packedBuffer = (float *)malloc(packedSize);

  float *A = _packedBuffer;
  float *B = A + MC * KC;

  _bufferA = Block { A, MR, (MC / MR) * KC, MR, false };
  _bufferB = Block { B, NR, (NC / NR) * KC, NR, false };
}

template<int MC, int KC, int NC, class TKernel>
inline GEMMCommon<MC, KC, NC, TKernel>::~GEMMCommon() {
  free(_packedBuffer);
  _packedBuffer = nullptr;
}


inline PackedBlock Pack(Block src, Block buf, int pack_size) {
  int numBlock = src.numCols / pack_size;
  int kc = src.numRows;
  PackedBlock tgt { buf.data, pack_size, kc, numBlock };
  ASSERT(pack_size * numBlock * kc <= buf.numCols * buf.numRows);

  for (int b = 0; b < numBlock; ++b) {
    Block srcBlock = src.sliceCol(b * pack_size, pack_size);
    Block tgtBlock = tgt.block(b);
    srcBlock.copyTo(tgtBlock);
  }

  int nc = src.numCols % pack_size;
  if (nc) {
    Block srcBlock = src.sliceCol(numBlock * pack_size, nc);
    Block tgtBlock = tgt.block(numBlock);
    tgtBlock.fillZero();

    tgtBlock = tgtBlock.sliceCol(0, nc);
    srcBlock.copyTo(tgtBlock);
    ++tgt.numBlocks;
  }

  return tgt;
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::split0ByNC() {
  int nb = _inputB.numCols / NC;
  int nc = _inputB.numCols % NC;

  for (int i = 0; i < nb; ++i) {
    Block Bn = _inputB.sliceCol(i * NC, NC);
    Block Cj = _inputC.sliceCol(i * NC, NC);
    split1ByKC(Bn, Cj);
  }

  if (nc) {
    Block Bn = _inputB.sliceCol(nb * NC, nc);
    Block Cj = _inputC.sliceCol(nb * NC, nc);
    split1ByKC(Bn, Cj);
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::split1ByKC(Block Bn, Block Cj) {
  int kb = Bn.numRows / KC;
  int kc = Bn.numRows % KC;

  for (int i = 0; i < kb; ++i) {
    Block Bkn = Bn.sliceRow(i * KC, KC);
    Block Ak = _inputA.sliceCol(i * KC, KC);
    PackedBlock Bp = nn::Pack(Bkn, _bufferB, NR);
    split2ByMC(Ak, Bp, Cj);
  }

  if (kc) {
    Block Bkn = Bn.sliceRow(kb * KC, kc);
    Block Ak = _inputA.sliceCol(kb * KC, kc);
    PackedBlock Bp = nn::Pack(Bkn, _bufferB, NR);
    split2ByMC(Ak, Bp, Cj);
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::split2ByMC(Block Ak, PackedBlock Bp, Block Cj) {
  int mb = Ak.numRows / MC;
  int mc = Ak.numRows % MC;

  for (int i = 0; i < mb; ++i) {
    Block Amk = Ak.sliceRow(i * MC, MC);
    Block Cij = Cj.sliceRow(i * MC, MC);
    PackedBlock Ap = nn::Pack(Amk.T(), _bufferA, MR);
    split3ByNR(Ap, Bp, Cij);
  }

  if (mc) {
    Block Amk = Ak.sliceRow(mb * MC, mc);
    Block Cij = Cj.sliceRow(mb * MC, mc);

    PackedBlock Ap = nn::Pack(Amk.T(), _bufferA, MR);
    split3ByNR(Ap, Bp, Cij); 
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::split3ByNR(PackedBlock Ap, PackedBlock Bp, Block Cij) {
  int np = Cij.numCols / NR;
  int nr = Cij.numCols % NR;

  #pragma omp parallel for num_threads(Environment::getCpuMathNumThreads())
  for (int i = 0; i < np; ++i) {
    Block Bpr = Bp.block(i);
    Block Cijn = Cij.sliceCol(i * NR, NR);
    split4ByMR(Ap, Bpr, Cijn);
  }

  if (nr) {
    Block Bpr = Bp.block(np);
    Block Cijn = Cij.sliceCol(np * NR, nr);
    split4ByMR(Ap, Bpr, Cijn);
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::split4ByMR(PackedBlock Ap, Block Bpr, Block Cijn) {
  int mp = Cijn.numRows / MR;
  int mr = Cijn.numRows % MR;

  for (int i = 0; i < mp; ++i) {
    Block Apr = Ap.block(i);
    Block Cijmn = Cijn.sliceRow(i * MR, MR);
    callKernel(Apr, Bpr, Cijmn);
  }

  if (mr) {
    Block Apr = Ap.block(mp);
    Block Cijmn = Cijn.sliceRow(mp * MR, mr);
    callKernel(Apr, Bpr, Cijmn);
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::callKernel(Block Apr, Block Bpr, Block Cijmn) {
  if (Cijmn.numRows < MR || Cijmn.numCols < NR) {
    float cData[MR * NR];
    Block cBuffer{cData, NR, MR, NR, false};

    cBuffer.fillZero();
    Block Cb = cBuffer.slice(0, 0, Cijmn.numRows, Cijmn.numCols);
    Cijmn.copyTo(Cb);

    TKernel::callKernel(Apr.numRows, Apr.data, Bpr.data, cBuffer.data, cBuffer.stride);
    Cb.copyTo(Cijmn);
  } else {
    TKernel::callKernel(Apr.numRows, Apr.data, Bpr.data, Cijmn.data, Cijmn.stride);
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::Apply(
    bool transa, bool transb,
    int m, int n, int k,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc) {
  _inputA = Block { (float *)A, lda, m, k, transa };
  _inputB = Block { (float *)B, ldb, k, n, transb };
  _inputC = Block { (float *)C, ldc, m, n, false };

  split0ByNC();
}

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
typedef GEMMCommon<576, 512, 4096, SGEMM12x32Avx512Kernel> SGEMMKernelAvx512;

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

std::unique_ptr<SGEMM> SGEMM::create() {
  switch (Environment::getCpuMathBackend()) {
    case CPUMathBackend::AVX512:
      return std::make_unique<SGEMMImplAvx512>();
    case CPUMathBackend::AVX2:
      return std::make_unique<SGEMMImplAvx2>();
    case CPUMathBackend::DEFAULT:
      return std::make_unique<SGEMMImplDefault>();
    default:
      NOT_IMPL();
  }
}

// -- class BatchSGEMM ----------

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

std::unique_ptr<BatchSGEMM> BatchSGEMM::create() {
  switch (Environment::getCpuMathBackend()) {
    case CPUMathBackend::AVX512:
      return std::make_unique<BatchSGEMMImpl<SGEMMImplAvx512>>();
    case CPUMathBackend::AVX2:
      return std::make_unique<BatchSGEMMImpl<SGEMMImplAvx2>>();
    case CPUMathBackend::DEFAULT:
      return std::make_unique<BatchSGEMMImpl<SGEMMImplDefault>>();
    default:
      NOT_IMPL();
  }
}

}  // namespace nn
}  // namespace llama
