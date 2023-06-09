#include "gemm.h"

#include <stdlib.h>
#include "gemm_common.h"
#include "gemm_kernel.h"
#include "log.h"
#include "util.h"

namespace llama {
namespace nn {

/*
struct GEMMConst {
  static constexpr int MC = 288;
  static constexpr int KC = 512;
  static constexpr int NC = 4096;
  static constexpr int MR = 6;
  static constexpr int NR = 16;
};
*/

struct GEMMConst {
  static constexpr int MC = 576;
  static constexpr int KC = 512;
  static constexpr int NC = 4096;
  static constexpr int MR = 12;
  static constexpr int NR = 32;
};


typedef GEMMCommon<288, 512, 4096, SGEMM6x16DefaultKernel> SGEMMFallback;
typedef GEMMCommon<288, 512, 4096, SGEMM6x16Avx2Kernel> SGEMMAvx2;
typedef GEMMCommon<576, 512, 4096, SGEMM6x16Avx512Kernel> SGEMMAvx512;

GEMM::GEMM() : _segmmBackend(GEMMBackend::DEFAULT) {
  chooseBackend();
}

void GEMM::chooseBackend() {
  if (util::isAvx512Available()) {
    _segmmBackend = GEMMBackend::AVX512;
  } else if (util::isAvx2Available()) {
    _segmmBackend = GEMMBackend::AVX2;
  } else {
    _segmmBackend = GEMMBackend::DEFAULT;
  }
}

void GEMM::sgemm(
    bool TransA, bool TransB, int M, int N, int K, const float *A, int lda,
    const float *B, int ldb, float *C, int ldc) {
  switch (_segmmBackend) {
    case GEMMBackend::AVX512:
      { SGEMMAvx512().Apply(TransA, TransB, M, N, K, A, lda, B, ldb, C, ldc); }
      break;
    case GEMMBackend::AVX2:
      { SGEMMAvx2().Apply(TransA, TransB, M, N, K, A, lda, B, ldb, C, ldc); }
      break;
    case GEMMBackend::DEFAULT:
      { SGEMMFallback().Apply(TransA, TransB, M, N, K, A, lda, B, ldb, C, ldc); }
      break;
    default:
      NOT_IMPL();
  }
}

}  // namespace nn
}  // namespace llama
