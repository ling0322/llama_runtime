#include "nn.h"

#include <stdlib.h>
#include <limits>
#include "nn.h"
#include "operators.h"
#include "gemm.h"
#include "tensor.h"

namespace llama {
namespace nn {

constexpr int kPrintEdgeItems = 3;

class CpuOperators : public Operators {
 public:
  Tensor Lookup(const Tensor &table, const Tensor &indices) override;
  Tensor LayerNorm(const Tensor &input) override;
  Tensor MatMul(const Tensor &a, const Tensor &b) override;
  Tensor Mul(const Tensor &input, float other)override;
  Tensor Softmax(const Tensor &input) override;
  Tensor Add(const Tensor &a, const Tensor &b) override;
  Tensor Tensor_(std::initializer_list<int> shape, DType dtype) override;
  Tensor Rand(std::initializer_list<int> shape, DType dtype) override;
  Tensor Zeros(std::initializer_list<int> shape, DType dtype) override;
  Tensor Contiguous(const Tensor &input) override;
  void Print(const Tensor &tensor) override;

 private:
  void Rand_Float32(Tensor *tensor);
  void Zeros_Float32(Tensor *tensor);
  void MatMul_Float32(const Tensor &A, const Tensor &B, Tensor *C);
  void Print1D_Float32(int d, const float *data);
  void Print2D_Float32(int m, int n, int lda, const float *data);
  void Print_Float32(const Tensor &tensor);
};

void CpuOperators::Rand_Float32(Tensor *tensor) {
  float *data = tensor->data<float>();
  int numel = tensor->numel();

  float randmax = RAND_MAX;
  for (int i = 0; i < numel; ++i) {
    data[i] = rand() / randmax;
  }
}

void CpuOperators::Zeros_Float32(Tensor *tensor) {
  float *data = tensor->data<float>();
  int numel = tensor->numel();

  float randmax = RAND_MAX;
  for (int i = 0; i < numel; ++i) {
    data[i] = 0.0f;
  }
}

void CpuOperators::MatMul_Float32(const Tensor &A, const Tensor &B, Tensor *C) {
  GEMM gemm;
  gemm.MatMul(
      A.shape(0), B.shape(1), A.shape(1),
      A.data<float>(), B.data<float>(), C->data<float>());
}

void CpuOperators::Print1D_Float32(int d, const float *data) {
  printf("[");
  for (int i = 0; i < d; ++i) {
    float elem = data[i];
    if (std::abs(elem) > 100 || std::abs(elem) < 0.01) {
      printf("%.4e", data[i]);
    } else {
      printf("%.4f", data[i]);
    }

    if (d > kPrintEdgeItems * 2 && i == kPrintEdgeItems - 1) {
      printf(" ... ");
      i += d - kPrintEdgeItems * 2;
    } else if (i < d - 1) {
      printf(", ");
    }
  }
  printf("]");
}

void CpuOperators::Print2D_Float32(int m, int n, int lda, const float *data) {
  printf("[");
  for (int i = 0; i < m; ++i) {
    if (i > 0) printf(" ");
    Print1D_Float32(n, data + i * lda);
    
    if (i < m - 1) printf(",\n"); 
    if (m > kPrintEdgeItems * 2 && i == kPrintEdgeItems - 1) {
      printf(" ...\n");
      i += m - kPrintEdgeItems * 2;
    }
  }
  printf("]");
}

void CpuOperators::Print_Float32(const Tensor &tensor) {
  int rank = tensor.rank();
  switch (rank) {
    case 1:
      Print1D_Float32(tensor.shape(0), tensor.data<float>());
      break;
    case 2:
      Print2D_Float32(tensor.shape(0), tensor.shape(1), tensor.stride(0),
                      tensor.data<float>());
      break;
    default:
      CHECK(false) << "unsupported rank for Print";
  }

  puts("");
}

Tensor CpuOperators::Tensor_(std::initializer_list<int> shape, DType dtype) {
  Tensor tensor;

  // rank
  tensor.FillShapeStride(shape);

  // data
  int numel = tensor.numel();
  tensor.data_ = std::make_shared<TensorData>(numel, dtype);

  return tensor;
}

Tensor CpuOperators::Rand(std::initializer_list<int> shape, DType dtype) {
  Tensor tensor = Tensor_(shape, dtype);
  switch (dtype) {
    case DType::kFloat:
      Rand_Float32(&tensor);
      break;
    default:
      CHECK(false) << "unsupported dtype for Rand";
  }

  return tensor;
}

Tensor CpuOperators::Zeros(std::initializer_list<int> shape, DType dtype) {
  Tensor tensor = Tensor_(shape, dtype);
  switch (dtype) {
    case DType::kFloat:
      Zeros_Float32(&tensor);
      break;
    default:
      CHECK(false) << "unsupported dtype for Zeros";
  }

  return tensor;
}

Tensor CpuOperators::MatMul(const Tensor &A, const Tensor &B) {
  CHECK(A.dtype() == B.dtype());
  CHECK(A.rank() == 2 && B.rank() == 2);
  CHECK(A.shape(1) == B.shape(0));

  Tensor C = Zeros({A.shape(0), B.shape(1)}, A.dtype());
  switch (A.dtype()) {
    case DType::kFloat:
      MatMul_Float32(A, B, &C);
      break;
    default:
      CHECK(false) << "unsupported dtype for MatMul";
  }

  return C;
}

void CpuOperators::Print(const Tensor &tensor) {
  switch (tensor.dtype()) {
    case DType::kFloat:
      Print_Float32(tensor);
      break;
    default:
      CHECK(false) << "unsupported dtype for Print";
  }
}

}  // namespace nn
}  // namespace llama
