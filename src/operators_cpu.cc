#include "nn.h"

#include <stdlib.h>
#include <cmath>
#include <limits>
#include <memory>
#include "nn.h"
#include "operators.h"
#include "gemm.h"
#include "tensor.h"

namespace llama {
namespace nn {

constexpr int kPrintEdgeItems = 3;

// the CPU implementation of Operators
class CpuOperators : public Operators {
 public:
  // Tensor Lookup(const Tensor &table, const Tensor &indices) override;
  // Tensor LayerNorm(const Tensor &input) override;
  Tensor MatMul(const Tensor &a, const Tensor &b) override;
  Tensor Mul(const Tensor &input, float other) override;
  Tensor Softmax(const Tensor &input) override;
  Tensor Add(const Tensor &a, const Tensor &b) override;
  Tensor Tensor_(std::initializer_list<int> shape, DType dtype) override;
  Tensor TensorLike(const Tensor &input) override;
  Tensor Rand(std::initializer_list<int> shape, DType dtype) override;
  Tensor Zeros(std::initializer_list<int> shape, DType dtype) override;
  // Tensor Contiguous(const Tensor &input) override;
  bool AllClose(const Tensor &A, const Tensor &B) override;
  void Print(const Tensor &tensor) override;

 private:
  typedef Tensor::Shape Shape;

  // apply elementwise binary operator to tensor `A` and tensor `B`, then store
  // result to tensor `C`
  //     C_ij = binary_op(A_ij, B_ij)
  // broadcast tensor `B` if necessary
  template<typename T>
  void ApplyBinaryOperator(
      util::Span<const Shape> shape_A,
      util::Span<const Shape> shape_B,
      util::Span<const Shape> shape_C,
      const T *data_A,
      const T *data_B,
      T *data_C,
      std::function<T(T input, T other)> binary_op);

  // apply unary 1D tensor operator on the last dimension of `A` and save
  // result to `C`. `C` should have the same shape as A.
  //   C_i = tensor1d_op(A_i) , here A_i is 1D tensor
  template<typename T>
  void ApplyUnary1DTensorOp(
      util::Span<const Shape> shape_A,
      util::Span<const Shape> shape_C,
      const T *data_A,
      T *data_C,
      std::function<void(Shape shape_A,
                         Shape shape_C,
                         const T *data_A,
                         T *data_C)> tensor1d_op);

  // call closure on all elements in a tensor.
  template<typename T>
  void ForEach(util::Span<const Shape> shape,
               const T *data,
               std::function<void(T v)> closure);

  // call closure on all elements in tensor A and tensor B. Shape of A should
  // equal to B.
  //   closure(A_ij, B_ij) , here A_ij and B_ij are scalar value
  template<typename T>
  void ForEach(util::Span<const Shape> shape_A,
               util::Span<const Shape> shape_B,
               const T *data_A,
               const T *data_B,
               std::function<void(T va, T vb)> closure);

  void Rand_Float32(Tensor *tensor);
  void Zeros_Float32(Tensor *tensor);
  void MatMul_Float32(const Tensor &A, const Tensor &B, Tensor *C);
  void Print1D_Float32(int d, const float *data);
  void Print2D_Float32(int m, int n, int lda, const float *data);
  void Print_Float32(const Tensor &tensor);
  void Add_Float32(const Tensor &A, const Tensor &B, Tensor *C);
  void Softmax_Float32(const Tensor &A, Tensor *C);
  bool AllClose_Float32(const Tensor &A,
                        const Tensor &B,
                        float rtol = 1e-05f,
                        float atol = 1e-08f);
  void Mul_Float32(const Tensor &A, float k, Tensor *C);
};

std::unique_ptr<Operators> CreateCpuOperators() {
  return std::make_unique<CpuOperators>();
}

void CpuOperators::Rand_Float32(Tensor *tensor) {
  float *data = tensor->data<float>();
  int64_t numel = tensor->numel();

  float randmax = RAND_MAX;
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = rand() / randmax;
  }
}

void CpuOperators::Zeros_Float32(Tensor *tensor) {
  float *data = tensor->data<float>();
  int64_t numel = tensor->numel();

  float randmax = RAND_MAX;
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = 0.0f;
  }
}

void CpuOperators::MatMul_Float32(const Tensor &A, const Tensor &B, Tensor *C) {
  GEMM gemm;
  
  bool transa, transb;
  int lda, ldb;
  if (A.stride(1) == 1) {
    transa = false;
    lda = A.stride(0);
  } else if (A.stride(0) == 1) {
    transa = true;
    lda = A.stride(1);
  } else {
    NOT_IMPL();
  }

  if (B.stride(1) == 1) {
    transb = false;
    ldb = B.stride(0);
  } else if (B.stride(0) == 1) {
    transb = true;
    ldb = B.stride(1);
  } else {
    NOT_IMPL();
  }

  gemm.MatMul(
      transa, transb,
      A.shape(0), B.shape(1), A.shape(1),
      A.data<float>(), lda,
      B.data<float>(), ldb,
      C->data<float>(), C->stride(0));
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

void CpuOperators::Add_Float32(const Tensor &A, const Tensor &B, Tensor *C) {
  *C = TensorLike(A);
  ApplyBinaryOperator<float>(
      util::MakeConstSpan(A.shape_),
      util::MakeConstSpan(B.shape_),
      util::MakeConstSpan(C->shape_),
      A.data<float>(),
      B.data<float>(),
      C->data<float>(),
      [](float input, float other) { return input + other; });
}

void CpuOperators::Softmax_Float32(const Tensor &A, Tensor *C) {
  auto softmax_op = [](Shape sa, Shape sc, const float *da, float *dc) {
    CHECK(sa.dimension == sc.dimension);

    double sum = 0;
    for (int i = 0; i < sa.dimension; ++i) {
      float va = da[i * sa.stride];
      sum += std::exp(va);
    }

    double logsum = std::log(sum);
    for (int i = 0; i < sa.dimension; ++i) {
      float va = da[i * sa.stride];
      float &vc = dc[i * sc.stride];
      vc = static_cast<float>(std::exp(va - logsum));
    }
  };

  ApplyUnary1DTensorOp<float>(
      util::MakeConstSpan(A.shape_),
      util::MakeConstSpan(C->shape_),
      A.data<float>(),
      C->data<float>(),
      softmax_op);
}

bool CpuOperators::AllClose_Float32(
    const Tensor &A,
    const Tensor &B,
    float rtol,
    float atol) {
  bool all_close = true;
  auto closure = [&all_close, rtol, atol](float va, float vb) {
    if (fabs(va - vb) > atol + rtol * fabs(vb)) {
      all_close = false;
    }
  };

  ForEach<float>(
      util::MakeConstSpan(A.shape_),
      util::MakeConstSpan(B.shape_),
      A.data<float>(),
      B.data<float>(),
      closure);
  
  return all_close;
}


void CpuOperators::Mul_Float32(const Tensor &A, float k, Tensor *C) {
  *C = TensorLike(A);
  ApplyUnary1DTensorOp<float>(
      util::MakeConstSpan(A.shape_),
      util::MakeConstSpan(C->shape_),
      A.data<float>(),
      C->data<float>(),
      [k](Shape sa, Shape sc, const float *da, float *dc) {
          CHECK(sa.dimension == sc.dimension);
          for (int i = 0; i < sa.dimension; ++i) {
            float va = da[i * sa.stride];
            float &vc = dc[i * sc.stride];
            vc = k * va;
          }
      });
}


template<typename T>
void CpuOperators::ApplyBinaryOperator(
      util::Span<const Shape> shape_A,
      util::Span<const Shape> shape_B,
      util::Span<const Shape> shape_C,
      const T *data_A,
      const T *data_B,
      T *data_C,
      std::function<T(T input, T other)> binary_op) {
  CHECK(shape_A.size() >= shape_B.size() && shape_B.size() >= 1);
  CHECK(shape_A.size() == shape_C.size());

  const Shape sa = shape_A.front();
  const Shape sb = shape_B.front();
  const Shape sc = shape_C.front();

  if (shape_A.size() > shape_B.size()) {
    // broadcast B
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
    CHECK(sa.dimension == sb.dimension && sa.dimension == sc.dimension);

    for (int i = 0; i < sa.dimension; ++i) {
      T va = data_A[i * sa.stride];
      T vb = data_B[i * sb.stride];
      T &vc = data_C[i * sc.stride];
      vc = binary_op(va, vb);
    }
  } else {
    NOT_IMPL();
  }
}

template<typename T>
void CpuOperators::ApplyUnary1DTensorOp(
    util::Span<const Shape> shape_A,
    util::Span<const Shape> shape_C,
    const T *data_A,
    T *data_C,
    std::function<void(Shape shape_A,
                       Shape shape_C,
                       const T *data_A,
                       T *data_C)> tensor1d_op) {
  CHECK(shape_A.size() == shape_C.size());
  CHECK(shape_A.size() >= 1);

  Shape sa = shape_A.front();
  Shape sc = shape_C.front();
  CHECK(sa.dimension == sc.dimension);

  if (shape_A.size() > 1) {
    for (int i = 0; i < sa.dimension; ++i) {
      const T *da = data_A + i * sa.stride;
      T *dc = data_C + i * sc.stride;
      ApplyUnary1DTensorOp(
          shape_A.subspan(1),
          shape_C.subspan(1),
          da,
          dc,
          tensor1d_op);
    }
  } else if (shape_A.size() == 1) {
    tensor1d_op(sa, sc, data_A, data_C);
  } else {
    NOT_IMPL();
  }
}

template<typename T>
void CpuOperators::ForEach(
    util::Span<const Shape> shape_A,
    util::Span<const Shape> shape_B,
    const T *data_A,
    const T *data_B,
    std::function<void(T va, T vb)> closure) {
    CHECK(shape_A.size() == shape_B.size());
  CHECK(shape_A.size() >= 1);

  Shape sa = shape_A.front();
  Shape sb = shape_B.front();
  CHECK(sa.dimension == sb.dimension);

  if (shape_A.size() > 1) {
    for (int i = 0; i < sa.dimension; ++i) {
      const T *da = data_A + i * sa.stride;
      const T *db = data_B + i * sb.stride;
      ForEach<T>(shape_A.subspan(1),
                 shape_B.subspan(1),
                 da,
                 db,
                 closure);
    }
  } else if (shape_A.size() == 1) {
    for (int i = 0; i < sa.dimension; ++i) {
      T va = data_A[i * sa.stride];
      T vb = data_B[i * sb.stride];
      closure(va, vb);
    }
  } else {
    NOT_IMPL();
  }
}

template<typename T>
void CpuOperators::ForEach(
    util::Span<const Shape> shape,
    const T *data,
    std::function<void(T v)> closure) {
  CHECK(shape.size() >= 1);

  Shape s = shape.front();
  if (shape.size() > 1) {
    for (int i = 0; i < s.dimension; ++i) {
      const T *d = data + i * s.stride;
      ForEach<T>(shape.subspan(1), d, closure);
    }
  } else if (shape.size() == 1) {
    for (int i = 0; i < s.dimension; ++i) {
      T v = data[i * s.stride];
      closure(v);
    }
  } else {
    NOT_IMPL();
  }
}

Tensor CpuOperators::Tensor_(std::initializer_list<int> shape, DType dtype) {
  Tensor tensor;

  int64_t numel = tensor.FillShapeStride(util::MakeConstSpan(shape));
  tensor.data_ = std::make_shared<TensorData>(numel, dtype);
  tensor.data_ptr_ = tensor.data_->data();

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
  int64_t numel = input.numel();
  tensor.data_ = std::make_shared<TensorData>(numel, input.dtype());
  tensor.data_ptr_ = tensor.data_->data();

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
  Tensor output;
  switch (input.dtype()) {
    case DType::kFloat:
      Add_Float32(input, other, &output);
      break;
    default:
      NOT_IMPL();
  }

  return output;
}

Tensor CpuOperators::Softmax(const Tensor &input) {
  Tensor output;
  switch (input.dtype()) {
    case DType::kFloat:
      Softmax_Float32(input, &output);
      break;
    default:
      NOT_IMPL();
  }

  return output;
}

bool CpuOperators::AllClose(const Tensor &A, const Tensor &B) {
  if (A.dtype() != B.dtype()) {
    return false;
  }

  switch (A.dtype()) {
    case DType::kFloat:
      return AllClose_Float32(A, B);
      break;
    default:
      NOT_IMPL();
  }
}

Tensor CpuOperators::Mul(const Tensor &A, float k) {
  Tensor C;

  switch (A.dtype()) {
    case DType::kFloat:
      Mul_Float32(A, k, &C);
      break;
    default:
      NOT_IMPL();
  }

  return C;
}

}  // namespace nn
}  // namespace llama
