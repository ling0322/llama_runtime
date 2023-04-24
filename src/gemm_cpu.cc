#include "gemm.h"

#include <stdlib.h>
#include "log.h"

namespace llama {
namespace nn {



GEMM::GEMM() : A_(nullptr), B_(nullptr), C_(nullptr) {
  int packed_size = (MC * KC + KC * NC + MR * NR) * sizeof(float);
  packed_buffer_ = (float *)malloc(packed_size);
  A_ = packed_buffer_;
  B_ = A_ + MC * KC;
  C_ = B_ + KC * NC;
}

GEMM::~GEMM() {
  free(packed_buffer_);
  packed_buffer_ = nullptr;
  A_ = nullptr;
  B_ = nullptr;
  C_ = nullptr;
}


void Pack(Block src, Block tgt, int pack_size) {
  int num_block = src.num_cols / pack_size;

  for (int b = 0; b < num_block; ++b) {
    Block src_block = src.ColRange(b * pack_size, pack_size);
    Block tgt_block = tgt.RowRange(b * src.num_cols, src.num_cols);
    src_block.CopyTo(tgt_block);
  }

  int _nc = src.num_cols % pack_size;
  if (_nc) {
    Block src_block = src.ColRange(num_block * pack_size, _nc);
    Block tgt_block = tgt.Range(
        num_block * src.num_cols,
        0,
        src.num_cols,
        _nc);
    src_block.CopyTo(tgt_block);
  }
}

void GEMM::Gemm5thLoopSplitByNC() {
  int nb = _B.num_cols / NC;
  int _nc = _B.num_cols % NC;

  for (int i = 0; i < nb; ++i) {
    Block Bn = _B.ColRange(i * NC, NC);
    Block Cj = _C.ColRange(i * NC, NC);
    Gemm4thLoopSplitByKC(Bn, Cj);
  }

  if (_nc) {
    Block Bn = _B.ColRange(nb * NC, _nc);
    Block Cj = _C.ColRange(nb * NC, _nc);
    Gemm4thLoopSplitByKC(Bn, Cj);
  }
}

void GEMM::Gemm4thLoopSplitByKC(Block Bn, Block Cj) {
  int kb = Bn.num_rows / KC;
  int _kc = Bn.num_rows % KC;

  for (int i = 0; i < kb; ++i) {
    Block Bkn = Bn.RowRange(i * KC, KC);
    Block Ak = _A.ColRange(i * KC, KC);
    nn::Pack(Bkn, _Bb, NR);
    Gemm3thLoopSplitByMC(Ak, _Bb, Cj);
  }

  if (_kc) {
    Block Bkn = Bn.RowRange(kb * KC, _kc);
    Block Ak = _A.ColRange(kb * KC, _kc);
    
    _Bb.FillZero();
    nn::Pack(Bkn, _Bb, NR);
    Gemm3thLoopSplitByMC(Ak, _Bb, Cj);
  }
}

void GEMM::Gemm3thLoopSplitByMC(Block Ak, Block Bp, Block Cj) {
  int mb = Ak.num_rows / MC;
  int _mc = Ak.num_rows % MC;

  for (int i = 0; i < mb; ++i) {
    Block Amk = Ak.RowRange(i * MC, MC);
    Block Cij = Cj.RowRange(i * MC, MC);
    nn::Pack(Amk.T(), _Ab, MR);
    Gemm2thLoopSplitByNR(_Ab, Bp, Cij);
  }

  if (_mc) {
    Block Amk = Ak.RowRange(mb * MC, _mc);
    Block Cij = Cj.RowRange(mb * MC, _mc);
    nn::Pack(Amk.T(), _Ab, MR);
    Gemm2thLoopSplitByNR(_Ab, Bp, Cij);
  }
}

void GEMM::Gemm2thLoopSplitByNR(Block Ap, Block Bp, Block Cij) {
  int np = Cij.num_cols / NR;
  int _nr = Cij.num_cols % NR;

  for (int i = 0; i < np; ++i) {
    Block Bpr = Bp.RowRange(i * KC, KC);
    Block Cijn = Cij.ColRange(i * NR, NR);
    Gemm1thLoopSplitByMR(Ap, Bpr, Cijn);
  }

  if (_nr) {
    Block Bpr = Bp.RowRange(np * KC, KC);
    Block Cijn = Cij.ColRange(np * NR, _nr);
    Gemm1thLoopSplitByMR(Ap, Bpr, Cijn);
  }
}

void GEMM::Gemm1thLoopSplitByMR(Block Ap, Block Bpr, Block Cijn) {
  int mp = Cijn.num_rows / MR;
  int _mr = Cijn.num_rows % MR;

  for (int i = 0; i < mp; ++i) {
    Block Apr = Ap.RowRange(i * KC, KC);
    Block Cijmn = Cijn.RowRange(i * MR, MR);
    CallMicroKernel(Apr, Bpr, Cijmn);
  }

  if (_mr) {
    Block Apr = Ap.RowRange(mp * KC, KC);
    Block Cijmn = Cijn.RowRange(mp * MR, _mr);
    CallMicroKernel(Apr, Bpr, Cijmn);
  }
}

void GEMM::CallMicroKernel(Block Apr, Block Bpr, Block Cijmn) {
  if (Cijmn.num_rows < MR || Cijmn.num_cols < NR) {
    MicroKernel(KC, Apr.data, Bpr.data, _Cb.data, _Cb.stride);
    Block Csrc = _Cb.Range(0, 0, Cijmn.num_rows, Cijmn.num_cols);
    Csrc.CopyTo(Cijmn);
  } else {
    MicroKernel(KC, Apr.data, Bpr.data, Cijmn.data, Cijmn.stride);
  }
}

void GEMM::MatMul(
    int m, int n, int k,
    const float *A,
    const float *B,
    float *C) {
  int mb = (m + MC - 1) / MC;
  int nb = (n + NC - 1) / NC;
  int kb = (k + KC - 1) / KC;

  int _mc = m % MC;
  int _nc = n % NC;
  int _kc = k % KC;

  int incRowA = k;
  int incRowB = n;
  int incRowC = n;

  Block Ab { (float *)A, k, m, k, false };
  Block Bb { (float *)B, n, k, n, false };
  Block Cb { (float *)C, n, m, n, false };

  Block A_buf { A_, MR, (MC / MR) * KC, MR, false };
  Block B_buf { B_, NR, (NC / NR) * KC, NR, false };

  int mc, nc, kc;
  int i, j, l;

  for (j = 0; j < nb; ++j) {
    nc = (j != nb - 1 || _nc == 0) ? NC : _nc;

    for (l = 0; l < kb; ++l) {
      kc = (l != kb - 1 || _kc == 0) ? KC: _kc;
      Block src = Bb.Range(l * KC, j * NC, kc, nc);
      nn::Pack(src, B_buf, NR);


      // PackB(kc, nc, &B[l * KC * incRowB + j * NC], incRowB, B_);
      for (i = 0; i < mb; ++i) {
        mc = (i != mb-1 || _mc == 0) ? MC : _mc;
        Block src = Ab.Range(i * MC, l * KC, mc, kc);
        nn::Pack(src.T(), A_buf, MR);

        // PackA(mc, kc, &A[i * MC * incRowA + l * KC], incRowA, A_);
        MacroKernel(mc, nc, kc, &C[i * MC * incRowC + j * NC], incRowC);
      }
    }
  }
}

void GEMM::PackMRxk(int k, const float *A, int incRowA, float *buffer) {
  int i, j;

  for (j = 0; j < k; ++j) {
    for (i = 0; i < MR; ++i) {
      buffer[i] = A[i * incRowA];
    }
    buffer += MR;
    ++A;
  }
}


void GEMM::PackA(int mc, int kc, const float *A, int incRowA, float *buffer) {
  int mp = mc / MR;
  int _mr = mc % MR;
  int i, j;

  for (i = 0; i < mp; ++i) {
    PackMRxk(kc, A, incRowA, buffer);
    buffer += kc * MR;
    A += MR * incRowA;
  }
  if (_mr > 0) {
    for (j = 0; j < kc; ++j) {
      for (i = 0; i < _mr; ++i) {
        buffer[i] = A[i * incRowA];
      }
      for (i = _mr; i < MR; ++i) {
        buffer[i] = 0.0;
      }
      buffer += MR;
      ++A;
    }
  }
}

void GEMM::PackkxNR(int k, const float *B, int incRowB, float *buffer) {
  int i, j;

  for (i = 0; i < k; ++i) {
    for (j = 0; j < NR; ++j) {
        buffer[j] = B[j];
    }
    buffer += NR;
    B += incRowB;
  }
}

void GEMM::PackB(
    int kc, int nc,
    const float *B, int incRowB,
    float *buffer) {
  int np  = nc / NR;
  int _nr = nc % NR;
  int i, j;

  for (j = 0; j < np; ++j) {
    PackkxNR(kc, B, incRowB, buffer);
    buffer += kc * NR;
    B += NR;
  }
  if (_nr > 0) {
    for (i = 0; i < kc; ++i) {
      for (j = 0; j < _nr; ++j) {
        buffer[j] = B[j];
      }
      for (j = _nr; j < NR; ++j) {
        buffer[j] = 0.0;
      }
      buffer += NR;
      B += incRowB;
    }
  }
}

void GEMM::MacroKernel(int mc, int nc, int kc, float *C, int incRowC) {
  int mp = (mc + MR - 1) / MR;
  int np = (nc + NR - 1) / NR;
  int _mr = mc % MR;
  int _nr = nc % NR;
  int mr, nr;
  int i, j;

  for (j = 0; j < np; ++j) {
    nr = (j != np - 1 || _nr == 0) ? NR : _nr;
    for (i = 0; i < mp; ++i) {
      mr = (i != mp - 1 || _mr == 0) ? MR : _mr;
      if (mr == MR && nr == NR) {
        MicroKernel(
          kc,
          &A_[i * kc * MR],
          &B_[j * kc * NR],
          &C[i * MR * incRowC + j * NR],
          incRowC);
      } else {
        for (int i = 0; i < MR * NR; ++i) {
            C_[i] = 0.0f;
        }
        MicroKernel(
            kc,
            &A_[i * kc * MR],
            &B_[j * kc * NR],
            C_, NR);
        float *Cb = &C[i * MR * incRowC + j * NR];
        for (int l = 0; l < mr; ++l) {
          for (int o = 0; o < nr; ++o) {
            Cb[l * incRowC + o] = C_[l * NR + o];
          }
        }
      }
    }
  }
}

void GEMM::MicroKernel(int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
  // a: MR x kc
  // b: kc x NR
  for (int k = 0; k < kc; ++k) {
    for (int m = 0; m < MR; ++m) {
      for (int n = 0; n < NR; ++n) {
          c[m * rs_c + n] += a[k * MR + m] * b[k * NR + n];
      }
    }
  }
}


}  // namespace nn
}  // namespace llama
