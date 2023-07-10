#pragma once

#include "pmpack/block.h"
#include "pmpack/pack.h"

namespace pmpack {

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
};



// -- class GEMMCommon ----------

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
    PackedBlock Bp = Pack(Bkn, _bufferB, NR);
    split2ByMC(Ak, Bp, Cj);
  }

  if (kc) {
    Block Bkn = Bn.sliceRow(kb * KC, kc);
    Block Ak = _inputA.sliceCol(kb * KC, kc);
    PackedBlock Bp = Pack(Bkn, _bufferB, NR);
    split2ByMC(Ak, Bp, Cj);
  }
}


template<class TMicroKernel>
void callGemmMicroKernel(Block A, Block B, Block C) {
  constexpr int MR = TMicroKernel::MR;
  constexpr int NR = TMicroKernel::NR;
  float dataCb[MR * NR];

  if (C.numRows < MR || C.numCols < NR) {
    Block Cb{dataCb, NR, MR, NR, false};
    Cb.fillZero();

    Block Cbs = Cb.slice(0, 0, C.numRows, C.numCols);
    C.copyTo(Cbs);

    TMicroKernel::callKernel(A.numRows, A.data, B.data, Cb.data, Cb.stride);
    Cbs.copyTo(C);
  } else {
    TMicroKernel::callKernel(A.numRows, A.data, B.data, C.data, C.stride);
  }
}

// GEMM macro-kernel: A(packed: MC, KC) DOT B(packed: KC, NC) -> C(MC, NC)
template<int MC, int KC, int NC, class TMicroKernel>
void applyGemmMacroKernel(PackedBlock A, PackedBlock B, Block C) {
  constexpr int MR = TMicroKernel::MR;
  constexpr int NR = TMicroKernel::NR;

  int np = (C.numCols + NR - 1) / NR;
  int mp = (C.numRows + MR - 1) / MR;
  int lastNr = C.numCols % NR;
  int lastMr = C.numRows % MR;

  #pragma omp parallel for num_threads(Environment::getCpuMathNumThreads()) schedule(dynamic,1)
  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < mp; ++j) {
      int nr = (i != np - 1 || lastNr == 0) ? NR : lastNr;
      int mr = (j != mp - 1 || lastMr == 0) ? MR : lastMr;

      Block Aj = A.block(j);
      Block Bi = B.block(i);
      Block Cji = C.slice(j * MR, i * NR, mr, nr);

      callGemmMicroKernel<TMicroKernel>(Aj, Bi, Cji);
    }
  }
}

template<int MC, int KC, int NC, class TKernel>
inline void GEMMCommon<MC, KC, NC, TKernel>::split2ByMC(Block Ak, PackedBlock Bp, Block Cj) {
  int mb = Ak.numRows / MC;
  int mc = Ak.numRows % MC;

  for (int i = 0; i < mb; ++i) {
    Block Amk = Ak.sliceRow(i * MC, MC);
    Block Cij = Cj.sliceRow(i * MC, MC);
    PackedBlock Ap = Pack(Amk.T(), _bufferA, MR);
    applyGemmMacroKernel<MC, KC, NC, TKernel>(Ap, Bp, Cij);
  }

  if (mc) {
    Block Amk = Ak.sliceRow(mb * MC, mc);
    Block Cij = Cj.sliceRow(mb * MC, mc);

    PackedBlock Ap = Pack(Amk.T(), _bufferA, MR);
    applyGemmMacroKernel<MC, KC, NC, TKernel>(Ap, Bp, Cij); 
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

}  // namespace pmpack

