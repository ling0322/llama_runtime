#include "nn.h"

#include <stdlib.h>
#include <limits>
#include <memory>
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
  Tensor TensorLike(const Tensor &input) override;
  Tensor Rand(std::initializer_list<int> shape, DType dtype) override;
  Tensor Zeros(std::initializer_list<int> shape, DType dtype) override;
  Tensor Contiguous(const Tensor &input) override;
  void Print(const Tensor &tensor) override;

 private:
  // apply elementwise binary operator to tensor `A` and tensor `B`, then store
  // result to tensor `C`
  //     C_ij = binary_op(A_ij, B_ij)
  // broadcast tensor `B` if necessary
  template<typename T>
  void ApplyBinaryOperator(
      util::Span<const Tensor::Shape> shape_A,
      util::Span<const Tensor::Shape> shape_B,
      util::Span<const Tensor::Shape> shape_C,
      const T *data_A,
      const T *data_B,
      T *data_C,
      std::function<T(T input, T other)> binary_op);

  void Rand_Float32(Tensor *tensor);
  void Zeros_Float32(Tensor *tensor);
  void MatMul_Float32(const Tensor &A, const Tensor &B, Tensor *C);
  void Print1D_Float32(int d, const float *data);
  void Print2D_Float32(int m, int n, int lda, const float *data);
  void Print_Float32(const Tensor &tensor);

  // Tensor(data_input) += Tensor(data_other)
  void Add_Float32(util::Span<Tensor::Shape> shape_input,
                   util::Span<Tensor::Shape> shape_other,
                   float *data_input,
                   float *data_other);
};

std::unique_ptr<Operators> CreateCpuOperators() {
  return std::make_unique<CpuOperators>();
}

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

template<typename T>
void CpuOperators::ApplyBinaryOperator(
      util::Span<const Tensor::Shape> shape_A,
      util::Span<const Tensor::Shape> shape_B,
      util::Span<const Tensor::Shape> shape_C,
      const T *data_A,
      const T *data_B,
      T *data_C,
      std::function<T(T input, T other)> binary_op) {
  CHECK(shape_A.size() >= shape_B.size() && shape_other.size() >= 1);
  CHECK(shape_A.size() == shape_C.size());

  if (shape_A.size() > shape_B.size()) {
    // broadcast B

    const Tensor::Shape sa = shape_A.front();
    const Tensor::Shape sc = shape_C.front();
    CHECK(sa.dimension == sc.dimension);
    
    for (int i = 0; i < sa.dimension; ++i) {
      const float *da = data_A + i * sa.stride;
      float *dc = data_C + i * sc.stride;
      ApplyBinaryOperator(
          shape_A.subspan(1),
          shape_B,
          shape_C.subspan(1),
          da,
          data_B,
          dc,
          binary_op);
    }
  } else if (shape_A.size() > 1) {
    // for n-D Tensor

    const Tensor::Shape sa = shape_A.front();
    const Tensor::Shape sb = shape_B.front();
    const Tensor::Shape sc = shape_C.front();
    CHECK(sa.dimension == sb.dimension && sa.dimension == sc.dimension);

    for (int i = 0; i < sa.dimension; ++i) {
      const float *da = data_A + i * sa.stride;
      const float *db = data_B + i * sb.stride;
      float *dc = data_C + i * sc.stride;
      ApplyBinaryOperator(
          shape_A.subspan(1),
          shape_B.subspan(1),
          shape_C.subspan(1),
          da,
          db,
          dc,
          binary_op);
    }
  } else if (shape_A.size() == 1) {
    // for 1-D tensor
    const Tensor::Shape sa = shape_A.front();
    const Tensor::Shape sb = shape_B.front();
    const Tensor::Shape sc = shape_C.front();
    CHECK(sa.dimension == sb.dimension && sa.dimension == sc.dimension);

    for (int i = 0; i < so.dimension; ++i) {
      T va = data_A[i * sa.stride];
      T vb = data_B[i * sb.stride];
      T &vc = data_C[i * sc.stride];
      vc = binary_op(va, vb);
    }
  } else {
    NOT_IMPL();
  }
}

Tensor CpuOperators::Tensor_(std::initializer_list<int> shape, DType dtype) {
  Tensor tensor;
  tensor.FillShapeStride(util::MakeConstSpan(shape));

  // data
  int numel = tensor.numel();
  tensor.data_ = std::make_shared<TensorData>(numel, dtype);

  return tensor;
}

Tensor CpuOperators::TensorLike(const Tensor &input) {
  std::vector<int> shape;
  for (const Tensor::Shape &s : input.shape_) {
    shape.push_back(s.dimension);
  }

  Tensor tensor;
  tensor.FillShapeStride(util::MakeConstSpan(shape));

  // data
  int numel = input.numel();
  tensor.data_ = std::make_shared<TensorData>(numel, input.dtype());

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

Tensor CpuOperators::Add(const Tensor &input, const Tensor &other) {
  Tensor output = TensorLike(input);
  ApplyBinaryOperator<float>(
      util::MakeConstSpan(input.shape_),
      util::MakeConstSpan(other.shape_),
      util::MakeConstSpan(output.shape_),
      input.data<float>(),
      other.data<float>(),
      output.data<float>(),
      [](float input, float other) { return input + other; });

  return output;
}

}  // namespace nn
}  // namespace llama
