#ifndef FASTALPACA_GEMM_COMMON_H_
#define FASTALPACA_GEMM_COMMON_H_

#include <stdint.h>
#include "log.h"

namespace llama {
namespace nn {

class SGEMM6x16DefaultKernel {
 public:
  static constexpr int MR = 6;
  static constexpr int NR = 16;

  static void GEMMKernel(
      int64_t kc, float *a, float *b, float *c, int64_t rs_c);
};

class SGEMM6x16Avx2Kernel {
 public:
  static constexpr int MR = 6;
  static constexpr int NR = 16;

  static void GEMMKernel(
      int64_t kc, float *a, float *b, float *c, int64_t rs_c);
};

class SGEMM6x16Avx512Kernel {
 public:
  static constexpr int MR = 12;
  static constexpr int NR = 32;

  static void GEMMKernel(
      int64_t kc, float *a, float *b, float *c, int64_t rs_c);
};


struct Block {
  float *data;
  int stride;
  int num_rows;
  int num_cols;
  bool transposed;

  constexpr Block RowRange(int row, int nr);
  constexpr Block ColRange(int col, int nc);
  constexpr Block Range(int row, int col, int nr, int nc);
  constexpr void CopyTo(Block tgt);
  constexpr Block T();
  constexpr void FillZero();
};

struct PackedBlock {
  float *data;
  int pack_size;
  int num_rows;
  int num_blocks;

  constexpr Block PackBlock(int i);
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
  float *packed_buffer_;

  void Loop0SplitByNC();
  void Loop1SplitByKC(Block Bn, Block Cj);
  void Loop2SplitByMC(Block Ak, PackedBlock Bp, Block Cj);
  void Loop3SplitByNR(PackedBlock Ap, PackedBlock Bp, Block Cij);
  void Loop4SplitByMR(PackedBlock Ap, Block Bpr, Block Cijn);
  void CallMicroKernel(Block Amkr, Block Bknr, Block Cijmn);

  Block _A;
  Block _B;
  Block _C;

  Block _Ab;
  Block _Bb;
  Block _Cb;
};

constexpr Block Block::RowRange(int row, int nr) {
  return Range(row, 0, nr, num_cols);
}
constexpr Block Block::ColRange(int col, int nc) {
  return Range(0, col, num_rows, nc);
}
constexpr Block Block::Range(int row, int col, int nr, int nc) {
  return Block {
    data + (transposed ? row + col * stride : row * stride + col),
    stride,
    nr,
    nc,
    transposed
  };
}
constexpr void Block::CopyTo(Block tgt) {
  ASSERT(num_rows == tgt.num_rows);
  ASSERT(num_cols == tgt.num_cols);

  for (int r = 0; r < num_rows; ++r) {
    for (int c = 0; c < num_cols; ++c) {
      if ((!transposed) && (!tgt.transposed)) {
        tgt.data[r * tgt.stride + c] = data[r * stride + c];
      } else if (transposed && (!tgt.transposed)) {
        tgt.data[r * tgt.stride + c] = data[r + c * stride];
      } else if ((!transposed) && tgt.transposed) {
        tgt.data[r + c * tgt.stride] = data[r * stride + c];
      } else if (transposed && tgt.transposed) {
        tgt.data[r + c * tgt.stride] = data[r + c * stride];
      }
    }
  }
}
constexpr Block Block::T() {
  return Block {
    data,
    stride,
    num_cols,
    num_rows,
    !transposed
  };
}
constexpr void Block::FillZero() {
  for (int r = 0; r < num_rows; ++r) {
    for (int c = 0; c < num_cols; ++c) {
      if (transposed) {
        data[r + c * stride] = 0.0f;
      } else {
        data[r * stride + c] = 0.0f;
      }
    }
  }
}
constexpr Block PackedBlock::PackBlock(int i) {
  return Block {
    data + pack_size * num_rows * i,
    pack_size,
    num_rows,
    pack_size,
    false
  };
}

template<int MC, int KC, int NC, class TKernel>
inline GEMMCommon<MC, KC, NC, TKernel>::GEMMCommon() {
  int packed_size = (MC * KC + KC * NC + MR * NR) * sizeof(float);
  packed_buffer_ = (float *)malloc(packed_size);

  float *A = packed_buffer_;
  float *B = A + MC * KC;
  float *C = B + KC * NC;

  _Ab = Block { A, MR, (MC / MR) * KC, MR, false };
  _Bb = Block { B, NR, (NC / NR) * KC, NR, false };
  _Cb = Block { C, NR, MR, NR, false };
}

template<int MC, int KC, int NC, class TKernel>
inline GEMMCommon<MC, KC, NC, TKernel>::~GEMMCommon() {
  free(packed_buffer_);
  packed_buffer_ = nullptr;
}


inline PackedBlock Pack(Block src, Block buf, int pack_size) {
  int num_block = src.num_cols / pack_size;
  int kc = src.num_rows;
  PackedBlock tgt { buf.data, pack_size, kc, num_block };
  ASSERT(pack_size * num_block * kc <= buf.num_cols * buf.num_rows);

  for (int b = 0; b < num_block; ++b) {
    Block src_block = src.ColRange(b * pack_size, pack_size);
    Block tgt_block = tgt.PackBlock(b);
    src_block.CopyTo(tgt_block);
  }

  int _nc = src.num_cols % pack_size;
  if (_nc) {
    Block src_block = src.ColRange(num_block * pack_size, _nc);
    Block tgt_block = tgt.PackBlock(num_block);
    tgt_block.FillZero();

    tgt_block = tgt_block.ColRange(0, _nc);
    src_block.CopyTo(tgt_block);
    ++tgt.num_blocks;
  }

  return tgt;
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::Loop0SplitByNC() {
  int nb = _B.num_cols / NC;
  int _nc = _B.num_cols % NC;

  for (int i = 0; i < nb; ++i) {
    Block Bn = _B.ColRange(i * NC, NC);
    Block Cj = _C.ColRange(i * NC, NC);
    Loop1SplitByKC(Bn, Cj);
  }

  if (_nc) {
    Block Bn = _B.ColRange(nb * NC, _nc);
    Block Cj = _C.ColRange(nb * NC, _nc);
    Loop1SplitByKC(Bn, Cj);
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::Loop1SplitByKC(
    Block Bn, Block Cj) {
  int kb = Bn.num_rows / KC;
  int _kc = Bn.num_rows % KC;

  for (int i = 0; i < kb; ++i) {
    Block Bkn = Bn.RowRange(i * KC, KC);
    Block Ak = _A.ColRange(i * KC, KC);
    PackedBlock Bp = nn::Pack(Bkn, _Bb, NR);
    Loop2SplitByMC(Ak, Bp, Cj);
  }

  if (_kc) {
    Block Bkn = Bn.RowRange(kb * KC, _kc);
    Block Ak = _A.ColRange(kb * KC, _kc);
    PackedBlock Bp = nn::Pack(Bkn, _Bb, NR);
    Loop2SplitByMC(Ak, Bp, Cj);
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::Loop2SplitByMC(
    Block Ak, PackedBlock Bp, Block Cj) {
  int mb = Ak.num_rows / MC;
  int _mc = Ak.num_rows % MC;

  for (int i = 0; i < mb; ++i) {
    Block Amk = Ak.RowRange(i * MC, MC);
    Block Cij = Cj.RowRange(i * MC, MC);
    PackedBlock Ap = nn::Pack(Amk.T(), _Ab, MR);
    Loop3SplitByNR(Ap, Bp, Cij);
  }

  if (_mc) {
    Block Amk = Ak.RowRange(mb * MC, _mc);
    Block Cij = Cj.RowRange(mb * MC, _mc);

    PackedBlock Ap = nn::Pack(Amk.T(), _Ab, MR);
    Loop3SplitByNR(Ap, Bp, Cij);
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::Loop3SplitByNR(
    PackedBlock Ap, PackedBlock Bp, Block Cij) {
  int np = Cij.num_cols / NR;
  int _nr = Cij.num_cols % NR;

  for (int i = 0; i < np; ++i) {
    Block Bpr = Bp.PackBlock(i);
    Block Cijn = Cij.ColRange(i * NR, NR);
    Loop4SplitByMR(Ap, Bpr, Cijn);
  }

  if (_nr) {
    Block Bpr = Bp.PackBlock(np);
    Block Cijn = Cij.ColRange(np * NR, _nr);
    Loop4SplitByMR(Ap, Bpr, Cijn);
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::Loop4SplitByMR(
    PackedBlock Ap, Block Bpr, Block Cijn) {
  int mp = Cijn.num_rows / MR;
  int _mr = Cijn.num_rows % MR;

  for (int i = 0; i < mp; ++i) {
    Block Apr = Ap.PackBlock(i);
    Block Cijmn = Cijn.RowRange(i * MR, MR);
    CallMicroKernel(Apr, Bpr, Cijmn);
  }

  if (_mr) {
    Block Apr = Ap.PackBlock(mp);
    Block Cijmn = Cijn.RowRange(mp * MR, _mr);
    CallMicroKernel(Apr, Bpr, Cijmn);
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::CallMicroKernel(
    Block Apr, Block Bpr, Block Cijmn) {
  if (Cijmn.num_rows < MR || Cijmn.num_cols < NR) {
    _Cb.FillZero();
    Block Cb = _Cb.Range(0, 0, Cijmn.num_rows, Cijmn.num_cols);
    Cijmn.CopyTo(Cb);

    MicroKernel(Apr.num_rows, Apr.data, Bpr.data, _Cb.data, _Cb.stride);
    Cb.CopyTo(Cijmn);
  } else {
    MicroKernel(Apr.num_rows, Apr.data, Bpr.data, Cijmn.data, Cijmn.stride);
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::Apply(
    bool transa, bool transb,
    int m, int n, int k,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc) {
  _A = Block { (float *)A, lda, m, k, transa };
  _B = Block { (float *)B, ldb, k, n, transb };
  _C = Block { (float *)C, ldc, m, n, false };

  Gemm1stLoopSplitByNC();
}

}  // namespace nn
}  // namespace llama

#endif  // FASTALPACA_GEMM_COMMON_H_
