// A default cpu-based GEMM

#ifndef GLM_RUNTIME_GLAS_H_
#define GLM_RUNTIME_GLAS_H_

#include <stdint.h>

namespace llama {
namespace nn {

struct Block {
  float *data;
  int stride;
  int num_rows;
  int num_cols;
  bool transposed;

  constexpr Block RowRange(int row, int nr) {
    return Range(row, 0, nr, num_cols);
  }

  constexpr Block ColRange(int col, int nc) {
    return Range(0, col, num_rows, nc);
  }

  constexpr Block Range(int row, int col, int nr, int nc) {
    return Block {
      data + (transposed ? row + col * stride : row * stride + col),
      stride,
      nr,
      nc,
      transposed
    };
  }

  constexpr void CopyTo(Block tgt) {
    CHECK(num_rows = tgt.num_rows);
    CHECK(num_cols == tgt.num_cols);

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

  constexpr Block T() {
    return Block {
      data,
      stride,
      num_cols,
      num_rows,
      !transposed
    };
  }

  constexpr void FillZero() {
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
};

struct GEMMConst {
  static constexpr int MC = 288;
  static constexpr int KC = 512;
  static constexpr int NC = 4096;
  static constexpr int MR = 6;
  static constexpr int NR = 16;
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
      int m, int n, int k,
      const float *A,
      const float *B,
      float *C);

  // Compute Y += alpha * X
  void Geaxpy(
      int m, int n,
      float alpha,
      const float *X, int incRowX,
      float *Y, int incRowY);

 private:
  float *packed_buffer_;
  float *A_;
  float *B_;
  float *C_;

  // Packing complete panels from A (i.e. without padding)
  void PackMRxk(int k, const float *A, int incRowA, float *buffer);
  
  // Packing panels from A with padding if required
  void PackA(
      int mc, int kc,
      const float *A, int incRowA,
      float *buffer);

  // Packing complete panels from B (i.e. without padding)
  void PackkxNR(int k, const float *B, int incRowB, float *buffer);

  // Packing panels from B with padding if required
  void PackB(
      int kc, int nc,
      const float *B, int incRowB,
      float *buffer);

  // Macro Kernel for the multiplication of blocks of A and B.  We assume that
  // these blocks were previously packed to buffers A_ and B_.
  void MacroKernel(int mc, int nc, int kc, float *C, int incRowC);

  // Computes C <- beta * C + alpha * A * B
  // Where A is MR * k, B is k * NR
  void MicroKernel(
      int64_t k,
      float *a,
      float *b,
      float *c, int64_t rs_c);

  void Gemm5thLoopSplitByNC();
  void Gemm4thLoopSplitByKC(Block Bn, Block Cj);
  void Gemm3thLoopSplitByMC(Block Ak, Block Bp, Block Cj);
  void Gemm2thLoopSplitByNR(Block Ap, Block Bp, Block Cij);
  void Gemm1thLoopSplitByMR(Block Ap, Block Bpr, Block Cijn);
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
