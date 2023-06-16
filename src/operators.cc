#include <stdlib.h>
#include <cmath>
#include <limits>
#include <memory>
#include "nn.h"
#include "operators.h"
#include "operators_cpu.h"

namespace llama {
namespace nn {

std::unique_ptr<Operators> Operators::create(Device device) {
  switch (device.getType()) {
    case Device::Type::kCpu:
      return CPUOperators::create();
    default:
      throw AbortedException("invalid device");
  }
}

// -- function Operators::matmul ----------

bool isColumnVector(const Tensor &x) {
  return x.getDim() == 2 && x.getShape(1) == 1;
}

bool isBatchedColumnVector(const Tensor &x) {
  return x.getDim() > 2 && x.getShape(-1) == 1;
}

bool isRowVector(const Tensor &x) {
  return x.getDim() == 2 && x.getShape(0) == 1;
}

bool isBatchedRowVector(const Tensor &x) {
  return x.getDim() > 2 && x.getShape(-2) == 1;
}

// batch row vector DOT matrix by GEMM operator.
Tensor brvXm2Gemm(Operators *F, const Tensor &A, const Tensor &B) {
  std::vector<Tensor::ShapeType> shape = A.getShape();
  Tensor A0 = A.view({-1, A.getShape(-1)});
  Tensor C0 = F->gemm(A0, B);

  shape.pop_back();
  shape.pop_back();
  shape.push_back(1);
  shape.push_back(B.getShape(1));

  return C0.view(shape);
}

// batch row vector DOT matrix by BMV operator.
Tensor brvXbmBmv(Operators *F, const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() >= B.getDim());

  Tensor B0 = B;
  while (B0.getDim() < A.getDim()) {
    B0 = B0.unsqueeze(0);
  }
  B0 = B0.transpose(-1, -2);

  Tensor A0 = A.squeeze(-2);
  Tensor C0 = F->bmv(B0, A0);
  std::vector<Tensor::ShapeType> shapeC = C0.getShape();
  int lastShape = shapeC.back();
  shapeC.pop_back();
  shapeC.push_back(1);
  shapeC.push_back(lastShape);

  return C0.view(shapeC);
}

// batch matrix DOT 2D matrix by GEMM operator.
Tensor bmXm2Gemm(Operators *F, const Tensor &A, const Tensor &B) {
  std::vector<int> shape;
  for (int d = 0; d < A.getDim() - 1; ++d) {
    shape.push_back(A.getShape(d));
  }

  Tensor C = F->gemm(A.view({-1, A.getShape(-1)}), B);
  shape.push_back(C.getShape(-1));
  return C.view(shape);
}

Tensor Operators::matmul(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() >= 2 && B.getDim() >= 2);

  if (A.getDim() == 2 && B.getDim() == 2) {
    if (isColumnVector(B)) {
      NOT_IMPL();
    } else if (isRowVector(A)) {
      NOT_IMPL();
    } else {
      return gemm(A, B);
    }
  } else if (A.getDim() > 2 && B.getDim() == 2) {
    if (isColumnVector(B) && !A.isContiguous()) {
      NOT_IMPL();
    } else if (isColumnVector(B) && A.isContiguous()) {
      NOT_IMPL();
    } else if (isBatchedRowVector(A) && A.isContiguous()) {
      return brvXm2Gemm(this, A, B);
    } else if (isBatchedRowVector(A) && !A.isContiguous()) {
      return brvXbmBmv(this, A, B);
    } else if (A.isContiguous()) {
      return bmXm2Gemm(this, A, B);
    } else {
      return bmm(A, B);
    }
  } else if (A.getDim() == 2 && B.getDim() > 2) {
    NOT_IMPL();
  } else if (A.getDim() > 2 && B.getDim() > 2) {
    if (isBatchedRowVector(A)) {
      return brvXbmBmv(this, A, B);
    } else {
      return bmm(A, B);
    }
  } else {
    NOT_IMPL();
  }
}


}  // namespace nn
}  // namespace llama
