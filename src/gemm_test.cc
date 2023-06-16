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

void testGEMM(int m, int k, int n, bool transa, bool transb) {
  auto F = Operators::create(Device::createForCPU());

  Tensor A = transa ? F->rand({k, m}, DType::kFloat)
                    : F->rand({m, k}, DType::kFloat);
  Tensor B = transb ? F->rand({n, k}, DType::kFloat)
                    : F->rand({k, n}, DType::kFloat);

  if (transa) A = A.transpose(0, 1);
  if (transb) B = B.transpose(0, 1);

  Tensor C = F->gemm(A, B);
  Tensor C_ref = RefMatMulFp32(A, B);

  REQUIRE(F->allClose(C, C_ref));
}

void testGEMV(int M, int N, bool TransA) {
  auto F = Operators::create(Device::createForCPU());

  Tensor A = TransA ? F->rand({N, M}, DType::kFloat) : F->rand({M, N}, DType::kFloat);
  Tensor x = F->rand({N, 1}, DType::kFloat);

  if (TransA) A = A.transpose(0, 1);

  Tensor C = F->gemv(A, x.view({N}));
  Tensor C_ref = RefMatMulFp32(A, x).view({M});

  F->print(C);
  F->print(C_ref);
  REQUIRE(F->allClose(C, C_ref));
}

int gemmTestShapes[][3] = {
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
  
  for (pshape = &gemmTestShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int k = (*pshape)[1];
    int n = (*pshape)[2];

    testGEMM(m, k, n, false, false);
    testGEMM(m, k, n, true, false);
    testGEMM(m, k, n, false, true);
  }
}

int gemvTestShapes[][2] = {
  {2, 8},
  {50, 10},
  {1, 1},
  {1024, 3}
};

TEST_CASE("float32 GEMV BVT", "[core][nn][gemm]") {
  int (*pshape)[2];
  
  for (pshape = &gemvTestShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int n = (*pshape)[1];

    testGEMV(m, n, false);
    testGEMV(m, n, true);
  }
}
