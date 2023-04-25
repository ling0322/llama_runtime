#include <functional>

#include "test_helper.h"
#include "gemm.h"
#include "nn.h"

using namespace llama;
using namespace nn;

Tensor RefMatMul_Float32(const Tensor &A, const Tensor &B) {
  REQUIRE(A.dtype() == B.dtype());
  REQUIRE(A.rank() == 2);
  REQUIRE(B.rank() == 2);
  REQUIRE(A.shape(1) == B.shape(0));
  REQUIRE(A.dtype() == DType::kFloat);

  Function F;
  Tensor C = F.Zeros({A.shape(0), B.shape(1)}, DType::kFloat);
  float *C_data = C.data<float>();
  const float *A_data = A.data<float>(),
              *B_data = B.data<float>();
  int lda = A.stride(0),
      ldb = B.stride(0),
      ldc = C.stride(0);

  for (int m = 0; m < A.shape(0); ++m) {
    for (int n = 0; n < B.shape(1); ++n) {
      for (int k = 0; k < A.shape(1); ++k) {
        C_data[ldc * m + n] += A_data[lda * m + k] * B_data[ldb * k + n];
      }
    }
  }

  return C;
}

bool AllClose2D_Float32(const Tensor &A,
                        const Tensor &B,
                        float rtol = 1e-05f,
                        float atol = 1e-08f) {
  if (A.rank() != B.rank()) return false;
  for (int d = 0; d < A.rank(); ++d) {
    if (A.shape(d) != B.shape(d)) return false;
  }

  const float *A_data = A.data<float>();
  const float *B_data = B.data<float>();

  int lda = A.stride(0);
  int ldb = B.stride(0);

  CHECK(A.rank() == 2);
  for (int m = 0; m < A.shape(0); ++m) {
    for (int n = 0; n < A.shape(1); ++n) {
      float a = A_data[m * lda + n];
      float b = B_data[m * ldb + n];
      if (fabs(a - b) > atol + rtol * b) return false;
    }
  }

  return true;
}

void TestGEMM(int m, int k, int n) {
  Function F;

  Tensor A = F.Rand({m, k}, DType::kFloat);
  Tensor B = F.Rand({k, n}, DType::kFloat);

  Tensor C = F.MatMul(A, B);
  Tensor C_ref = RefMatMul_Float32(A, B);

  REQUIRE(AllClose2D_Float32(C, C_ref));
}

TEST_CASE("float32 GEMM BVT", "[core][nn]") {
  TestGEMM(1, 1, 1);
  TestGEMM(2, 2, 2);
  TestGEMM(50, 50, 1);
  TestGEMM(513, 2, 513);
  TestGEMM(16, 16, 5000);
  TestGEMM(16, 5000, 16);
  TestGEMM(5000, 16, 16);
}
