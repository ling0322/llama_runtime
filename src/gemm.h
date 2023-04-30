// A default cpu-based GEMM

#ifndef GLM_RUNTIME_GLAS_H_
#define GLM_RUNTIME_GLAS_H_

#include <stdint.h>
#include "log.h"

namespace llama {
namespace nn {

struct GEMMConst {
  static constexpr int MC = 288;
  static constexpr int KC = 512;
  static constexpr int NC = 4096;
  static constexpr int MR = 6;
  static constexpr int NR = 16;
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

class GEMM {
 public:

  GEMM();
  ~GEMM();

  static constexpr int MC = GEMMConst::MC;
  static constexpr int KC = GEMMConst::KC;
  static constexpr int NC = GEMMConst::NC;
  static constexpr int MR = GEMMConst::MR;
  static constexpr int NR = GEMMConst::NR;

  // Compute C <- A * B
  void MatMul(
      bool transa, bool transb,
      int m, int n, int k,
      const float *A, int lda,
      const float *B, int ldb,
      float *C, int ldc);

 private:
  float *packed_buffer_;

  // Computes C <- beta * C + alpha * A * B
  // Where A is MR * k, B is k * NR
  void MicroKernel(
      int64_t k,
      float *a,
      float *b,
      float *c, int64_t rs_c);

  void Gemm5thLoopSplitByNC();
  void Gemm4thLoopSplitByKC(Block Bn, Block Cj);
  void Gemm3thLoopSplitByMC(Block Ak, PackedBlock Bp, Block Cj);
  void Gemm2thLoopSplitByNR(PackedBlock Ap, PackedBlock Bp, Block Cij);
  void Gemm1thLoopSplitByMR(PackedBlock Ap, Block Bpr, Block Cijn);
  void CallMicroKernel(Block Amkr, Block Bknr, Block Cijmn);

  Block _A;
  Block _B;
  Block _C;

  Block _Ab;
  Block _Bb;
  Block _Cb;
};

}  // namespace nn
}  // namespace llama

#endif  // GLM_RUNTIME_GLAS_H_
