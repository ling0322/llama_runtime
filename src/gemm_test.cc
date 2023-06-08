#include <functional>

#include "test_helper.h"
#include "gemm.h"
#include "nn.h"
#include "operators.h"

using namespace llama;
using namespace nn;

Tensor RefMatMulFp32(const Tensor &A, const Tensor &B) {
  REQUIRE(A.getDType() == B.getDType());
  REQUIRE(A.getDim() == 2);
  REQUIRE(B.getDim() == 2);
  REQUIRE(A.getShape(1) == B.getShape(0));
  REQUIRE(A.getDType() == DType::kFloat);

  auto F = Operators::create(Device::createForCPU());

  Tensor C = F->zeros({A.getShape(0), B.getShape(1)}, DType::kFloat);
  float *dataC = C.getData<float>();
  const float *dataA = A.getData<float>(),
              *dataB = B.getData<float>();
  int stride0A = A.getStride(0);
  int stride1A = A.getStride(1);
  int stride0B = B.getStride(0);
  int stride1B = B.getStride(1);
  int ldc = C.getStride(0);

  for (int m = 0; m < A.getShape(0); ++m) {
    for (int n = 0; n < B.getShape(1); ++n) {
      for (int k = 0; k < A.getShape(1); ++k) {
        float va = dataA[stride0A * m + k * stride1A];
        float vb = dataB[stride0B * k + n * stride1B];
        dataC[ldc * m + n] += va * vb;
      }
    }
  }

  return C;
}

void TestGEMM(int m, int k, int n, bool transa, bool transb) {
  auto F = Operators::create(Device::createForCPU());

  Tensor A = transa ? F->rand({k, m}, DType::kFloat)
                    : F->rand({m, k}, DType::kFloat);
  Tensor B = transb ? F->rand({n, k}, DType::kFloat)
                    : F->rand({k, n}, DType::kFloat);

  if (transa) A = A.transpose(0, 1);
  if (transb) B = B.transpose(0, 1);

  Tensor C = F->matmul(A, B);
  Tensor C_ref = RefMatMulFp32(A, B);

  REQUIRE(F->allClose(C, C_ref));
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
