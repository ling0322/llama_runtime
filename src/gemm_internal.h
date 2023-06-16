#ifndef FASTALPACA_GEMM_KERNEL_H_
#define FASTALPACA_GEMM_KERNEL_H_

#include <stdint.h>
#include "log.h"

namespace llama {
namespace nn {


class SGEMM6x16DefaultKernel {
 public:
  static constexpr int MR = 6;
  static constexpr int NR = 16;
  static void callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c);
};

class SGEMM6x16Avx2Kernel {
 public:
  static constexpr int MR = 6;
  static constexpr int NR = 16;
  static void callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c);
};

class SGEMM6x16Avx512Kernel {
 public:
  static constexpr int MR = 12;
  static constexpr int NR = 32;
  static void callKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c);
};

class SAXPYAvx2Kernel {
 public:
  static void callKernel(int64_t n, float a, const float *x, float *y);
};

class SDOTAvx2Kernel {
 public:
  static float callKernel(int64_t n, const float *x, const float *y);
};

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
  Block _bufferC;

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
  int packedSize = (MC * KC + KC * NC + MR * NR) * sizeof(float);
  _packedBuffer = (float *)malloc(packedSize);

  float *A = _packedBuffer;
  float *B = A + MC * KC;
  float *C = B + KC * NC;

  _bufferA = Block { A, MR, (MC / MR) * KC, MR, false };
  _bufferB = Block { B, NR, (NC / NR) * KC, NR, false };
  _bufferC = Block { C, NR, MR, NR, false };
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
    _bufferC.fillZero();
    Block Cb = _bufferC.slice(0, 0, Cijmn.numRows, Cijmn.numCols);
    Cijmn.copyTo(Cb);

    TKernel::callKernel(Apr.numRows, Apr.data, Bpr.data, _bufferC.data, _bufferC.stride);
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

}  // namespace nn
}  // namespace llama

#endif  // FASTALPACA_GEMM_KERNEL_H_
