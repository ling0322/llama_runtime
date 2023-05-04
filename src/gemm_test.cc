#include <functional>

#include "test_helper.h"
#include "gemm.h"
#include "nn.h"
#include "operators.h"

using namespace llama;
using namespace nn;

Tensor RefMatMul_Float32(const Tensor &A, const Tensor &B) {
  REQUIRE(A.dtype() == B.dtype());
  REQUIRE(A.rank() == 2);
  REQUIRE(B.rank() == 2);
  REQUIRE(A.shape(1) == B.shape(0));
  REQUIRE(A.dtype() == DType::kFloat);

  auto F = Operators::FromDevice(Device::CPU());

  Tensor C = F->Zeros({A.shape(0), B.shape(1)}, DType::kFloat);
  float *C_data = C.data<float>();
  const float *A_data = A.data<float>(),
              *B_data = B.data<float>();
  int stride0_A = A.stride(0);
  int stride1_A = A.stride(1);
  int stride0_B = B.stride(0);
  int stride1_B = B.stride(1);
  int ldc = C.stride(0);

  for (int m = 0; m < A.shape(0); ++m) {
    for (int n = 0; n < B.shape(1); ++n) {
      for (int k = 0; k < A.shape(1); ++k) {
        float va = A_data[stride0_A * m + k * stride1_A];
        float vb = B_data[stride0_B * k + n * stride1_B];
        C_data[ldc * m + n] += va * vb;
      }
    }
  }

  return C;
}

void TestGEMM(int m, int k, int n, bool transa, bool transb) {
  auto F = Operators::FromDevice(Device::CPU());

  Tensor A = transa ? F->Rand({k, m}, DType::kFloat)
                    : F->Rand({m, k}, DType::kFloat);
  Tensor B = transb ? F->Rand({n, k}, DType::kFloat)
                    : F->Rand({k, n}, DType::kFloat);

  if (transa) A = A.Transpose(0, 1);
  if (transb) B = B.Transpose(0, 1);

  Tensor C = F->MatMul(A, B);
  Tensor C_ref = RefMatMul_Float32(A, B);

  REQUIRE(F->AllClose(C, C_ref));
}

int test_shapes[][3] = {
  {50, 50, 1},
  {1, 1, 1},
  {2, 2, 2},
  {50, 50, 1},
  {513, 2, 513},
  {16, 16, 5000},
  {16, 5000, 16},
  {5000, 16, 16},
  {0, 0, 0}
};

TEST_CASE("float32 GEMM BVT", "[core][nn][gemm]") {
  int (*pshape)[3];
  
  for (pshape = &test_shapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int k = (*pshape)[1];
    int n = (*pshape)[2];

    TestGEMM(m, k, n, false, false);
    TestGEMM(m, k, n, true, false);
    TestGEMM(m, k, n, false, true);
  }
}
