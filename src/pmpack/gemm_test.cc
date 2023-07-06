#include "common/test_helper.h"

#include "pmpack/gemm_kernel.h"
#include "pmpack/pmpack.h"
#include "nn/nn_test_helper.h"
#include "nn/operators.h"
#include "util/random.h"
#include "util/util.h"

using namespace llama;
using namespace nn;

constexpr uint32_t MagicNumber = 0x55aa;

void refGemmFp32QInt8Fp32(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const void *B,
    const float *scaleDataB,
    int groupSizeB,
    float *C,
    int ldc) {
  CHECK(groupSizeB == K);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      const float *Aj = A + j * lda;
      const uint8_t *Bi = reinterpret_cast<const uint8_t *>(B) + i * K / 2;
      C[j * ldc + i] = DOTFp32Int4Fp32FallbackKernel::apply(K, Aj, Bi, scaleDataB[i]);
    }
  }
}

bool isClose(util::Span<const float> A, util::Span<const float> B) {
  if (A.size() != B.size()) 
    return false;

  for (int i = 0; i < A.size(); ++i) {
    if (fabs(A[i] - B[i]) > 1e-4)
      return false;
  }

  return true;
}

void testGemmFp32QInt4Fp32(int M, int N, int K) {
  std::vector<float> A(M * K);
  std::vector<uint8_t> B(K * N / 2);
  std::vector<float> scaleB(N);

  util::Random random(MagicNumber);

  random.fill(util::makeSpan(A));
  random.fillUInt8(util::makeSpan(B));
  random.fill(util::makeSpan(scaleB));

  std::vector<float> C(M * N);
  std::vector<float> refC(M * N);

  refGemmFp32QInt8Fp32(
      false,
      true,
      M,
      N,
      K,
      A.data(),
      K,
      B.data(),
      scaleB.data(),
      K,
      refC.data(),
      N);

  pmpack_gemm_fp32qint4fp32(
      false,
      true,
      M,
      N,
      K,
      A.data(),
      K,
      B.data(),
      scaleB.data(),
      K,
      C.data(),
      N);

  REQUIRE(isClose(C, refC));
}

TEST_CASE("test dotFp32Int4Fp32", "[core][gemm][avx2]") {
  constexpr int DIM = 1024;

  std::vector<float> x(DIM);
  std::vector<uint8_t> y(DIM / 2);

  util::Random random(MagicNumber);

  random.fill(util::makeSpan(x));
  random.fillUInt8(util::makeSpan(y));

  float rs = DOTFp32Int4Fp32FallbackKernel::apply(DIM, x.data(), y.data(), 0.1f);
  float s = DOTFp32Int4Fp32Avx2Kernel::apply(DIM, x.data(), y.data(), 0.1f);

  REQUIRE(fabs(rs - s) < 1e-5);
}

TEST_CASE("test pmpack_gemm_fp32qint4fp32", "[core][gemm][avx2]") {
  testGemmFp32QInt4Fp32(32, 32, 32);
  testGemmFp32QInt4Fp32(1, 32, 32);
}
