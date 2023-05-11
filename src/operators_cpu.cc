#include "operators_cpu.h"

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
class CpuOperators::Impl {
 public:
  typedef Size::Elem Shape;

  // sub-tensor. Stores the shape and data pointer to a sub region of the 
  // original Tensor. It's faster than Tensor when passing as parameters.
  template<typename T>
  struct SubTensor;
  typedef SubTensor<float> Subtensorf;
  typedef SubTensor<const LongType> SubtensorCl;
  typedef SubTensor<const float> SubtensorCf;

  // type for lambda expression with 1D tensor as input. NR means no return.
  template<typename T>
  using UnaryOp = std::function<void(
      SubTensor<const T> A,
      SubTensor<T> C)>;
  template<typename T>
  using BinaryOp = std::function<void(
      SubTensor<const T> A,
      SubTensor<const T> B,
      SubTensor<T> C)>;
  template<typename T>
  using BinaryOpNR = std::function<void(
      SubTensor<const T> A,
      SubTensor<const T> B)>;


  // apply elementwise binary operator to tensor `A` and tensor `B`, then store
  // result to tensor `C`
  //     C_i = binary_op(A_i, B_i) , here A_i is 1D tensor
  // broadcast tensor `B` if necessary
  template<typename T>
  void ApplyBinaryOperator(
      SubTensor<const T> A,
      SubTensor<const T> B,
      SubTensor<T> C,
      BinaryOp<T> binary_op);

  // apply unary 1D tensor operator on the last dimension of `A` and save
  // result to `C`. `C` should have the same shape as A.
  //   C_i = tensor1d_op(A_i) , here A_i is 1D tensor
  template<typename T>
  void ApplyUnary1DTensorOp(
      SubTensor<const T> A,
      SubTensor<T> C,
      UnaryOp<T> unary_op);

  // call closure on all elements in tensor A and tensor B. Shape of A should
  // equal to B.
  //   closure(A_i, B_i) , here A_i and B_i are 1D tensor.
  template<typename T>
  void ForEach(
      SubTensor<const T> A,
      SubTensor<const T> B,
      BinaryOpNR<T> binary_op);

  // Get sub-tensor from Tensor.
  template<typename T>
  SubTensor<T> MakeSubTensor(Tensor &tensor); 
  template<typename T>
  SubTensor<const T> MakeConstSubTensor(const Tensor &tensor); 

  Tensor Tensor_(util::Span<const int> shape, DType dtype);
  Tensor TensorLike(SubtensorCf A);

  Tensor MatMul_Float32(SubtensorCf A, SubtensorCf B);

  Tensor Mul_Float32(SubtensorCf A, float k);
  Tensor Add_Float32(SubtensorCf A, SubtensorCf B);
  Tensor Softmax_Float32(SubtensorCf A);

  void Rand_Float32(Tensor *tensor);
  void Zeros_Float32(Subtensorf tensor);
  void Print1D_Float32(SubtensorCf A);
  void PrintND_Float32(SubtensorCf A, int pad_space);
  void Print_Float32(SubtensorCf tensor);
  
  bool AllClose_Float32(SubtensorCf A, SubtensorCf B, float rtol, float atol);
  void GEMM_Float32(SubtensorCf A, SubtensorCf B, Subtensorf C);
  void BMM_Float32(SubtensorCf A, SubtensorCf B, Subtensorf C);

  // Copy data from src to tgt. tgt and src should have the same dtype and
  // shape
  void Copy_Float32(SubtensorCf src, Subtensorf tgt);

  Tensor Lookup_Float32(SubtensorCf table, SubtensorCl indices);
  Tensor LayerNorm_Float32(
      SubtensorCf input,
      SubtensorCf weight,
      SubtensorCf bias,
      float eps);
};

// ---------------------------------------------------------------------------+
// class CpuOperators::Impl::SubTensor                                        |
// ---------------------------------------------------------------------------+

// sub-tensor. Stores the shape and data pointer to a sub region of the original
// Tensor. It's faster than Tensor when passing as parameters.
template<typename T>
struct CpuOperators::Impl::SubTensor {
  util::Span<const Shape> shape;
  T *data;

  // get sub-tensor of this SubTensor.
  SubTensor<T> subtensor(int index) {
    return SubTensor<T>{
      this->shape.subspan(1),
      data + index * this->shape[0].stride
    };
  }

  // get dimension or stride for an axis. NOTE: negative index is NOT supported.
  int dimension(int index) { return shape[index].shape; }
  int stride(int index) { return shape[index].stride; }

  // get element from 1D sub-tensor. NOTE: elem() will not check the shape.
  T &elem(int index) {
    return data[index * this->shape[0].stride];
  }

  // number of element.
  int64_t numel() {
    int64_t n = 1;
    for (const Shape &s : shape) {
      n *= s.shape;
    }
    return n;
  }

  // get rank.
  int rank() { return shape.size(); }
};

template<typename T>
inline auto CpuOperators::Impl::MakeSubTensor(Tensor &tensor) -> SubTensor<T> {
  return SubTensor<T>{
    util::MakeConstSpan(tensor.size_.data_),
    tensor.data<T>(),
  };
} 

template<typename T>
inline auto CpuOperators::Impl::MakeConstSubTensor(
    const Tensor &tensor) -> SubTensor<const T> {
  return SubTensor<const T>{
    util::MakeConstSpan(tensor.size_.data_),
    tensor.data<T>(),
  };
} 

// ---------------------------------------------------------------------------+
// CpuOperators::Impl                                                         |
// ---------------------------------------------------------------------------+

template<typename T>
void CpuOperators::Impl::ApplyBinaryOperator(
    SubTensor<const T> A,
    SubTensor<const T> B,
    SubTensor<T> C,
    BinaryOp<T> binary_op) {
  CHECK(A.rank() >= B.rank() && B.rank() >= 1);
  CHECK(A.rank() == C.rank());

  if (A.rank() > B.rank()) {
    // broadcast B
    CHECK(A.dimension(0) == C.dimension(0));

    for (int i = 0; i < A.dimension(0); ++i) {
      ApplyBinaryOperator(A.subtensor(i), B, C.subtensor(i), binary_op);
    }
  } else if (A.rank() > 1) {
    // for n-D Tensor
    CHECK(A.dimension(0) == C.dimension(0) && A.dimension(0) == B.dimension(0));

    for (int i = 0; i < A.dimension(0); ++i) {
      ApplyBinaryOperator(
          A.subtensor(i),
          B.subtensor(i),
          C.subtensor(i),
          binary_op);
    }
  } else if (A.rank() == 1) {
    // for 1-D tensor
    CHECK(A.dimension(0) == C.dimension(0) && A.dimension(0) == B.dimension(0));
    binary_op(A, B, C);
  } else {
    NOT_IMPL();
  }
}

template<typename T>
void CpuOperators::Impl::ApplyUnary1DTensorOp(
    SubTensor<const T> A,
    SubTensor<T> C,
    UnaryOp<T> unary_op) {
  CHECK(A.rank() == C.rank());
  CHECK(A.rank() >= 1);

  CHECK(A.dimension(0) == C.dimension(0));
  if (A.rank() > 1) {
    for (int i = 0; i < A.dimension(0); ++i) {
      ApplyUnary1DTensorOp(A.subtensor(i), C.subtensor(i), unary_op);
    }
  } else if (A.rank() == 1) {
    unary_op(A, C);
  } else {
    NOT_IMPL();
  }
}

template<typename T>
void CpuOperators::Impl::ForEach(
    SubTensor<const T> A,
    SubTensor<const T> B,
    BinaryOpNR<T> closure) {
  CHECK(A.rank() == B.rank());
  CHECK(A.rank() >= 1);

  CHECK(A.dimension(0) == B.dimension(0));
  if (A.rank() > 1) {
    for (int i = 0; i < A.dimension(0); ++i) {
      ForEach<T>(A.subtensor(i), B.subtensor(i), closure);
    }
  } else if (A.rank() == 1) {
    closure(A, B);
  } else {
    NOT_IMPL();
  }
}

Tensor CpuOperators::Impl::Tensor_(util::Span<const int> shape, DType dtype) {
  Tensor tensor;

  tensor.size_ = Size(util::MakeConstSpan(shape));
  int64_t numel = tensor.size_.numel();

  tensor.data_ = std::make_shared<TensorData>(numel, dtype);
  tensor.data_ptr_ = tensor.data_->data();

  return tensor;
}

void CpuOperators::Impl::Rand_Float32(Tensor *tensor) {
  float *data = tensor->data<float>();
  int64_t numel = tensor->numel();

  float randmax = RAND_MAX;
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = rand() / randmax;
  }
}

void CpuOperators::Impl::Zeros_Float32(Subtensorf tensor) {
  // make sure tensor is contiguous.
  CHECK(tensor.numel() == tensor.stride(0) * tensor.dimension(0));

  float *data = tensor.data;
  int64_t numel = tensor.numel();

  for (int64_t i = 0; i < numel; ++i) {
    data[i] = 0.0f;
  }
}

Tensor CpuOperators::Impl::MatMul_Float32(SubtensorCf A, SubtensorCf B) {
  CHECK(A.rank() >= B.rank() && B.rank() >= 2);

  if (A.rank() == B.rank() && A.rank() == 2) {
    // Both A and B is 2D tensor, call GEMM
    Tensor C = Tensor_(util::MakeConstSpan({A.dimension(0), B.dimension(1)}),
                       DType::kFloat);
    Subtensorf Cs = MakeSubTensor<float>(C);
    Zeros_Float32(Cs);
    GEMM_Float32(A, B, Cs);

    return C;
  } else {
    // BMM
    std::vector<int> shape;

    // broadcast B
    int broadcast_dims = A.rank() - B.rank();
    for (int i = 0; i < broadcast_dims; ++i) {
      shape.push_back(A.dimension(i));
    }

    // batch dim: B.shape(i) == A.shape(broadcast_dims + i)
    int batch_dims = B.rank() - 2;
    for (int i = 0; i < batch_dims; ++i) {
      CHECK(A.dimension(broadcast_dims + i) == B.dimension(i));
      shape.push_back(B.dimension(i));
    }

    // GEMM dims (A.shape(-2), B.shape(-1))
    shape.push_back(A.dimension(broadcast_dims + batch_dims));
    shape.push_back(B.dimension(batch_dims + 1));

    Tensor C = Tensor_(util::MakeConstSpan(shape), DType::kFloat);
    Subtensorf Cs = MakeSubTensor<float>(C);
    Zeros_Float32(Cs);

    BMM_Float32(A, B, Cs);
    return C;
  }
}

void CpuOperators::Impl::GEMM_Float32(
    SubtensorCf A,
    SubtensorCf B,
    Subtensorf C) {
  CHECK(A.rank() == B.rank() && A.rank() == C.rank());
  CHECK(A.rank() == 2);
  CHECK(A.dimension(0) == C.dimension(0));
  CHECK(A.dimension(1) == B.dimension(0));
  CHECK(B.dimension(1) == C.dimension(1));

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

  int m = A.dimension(0);
  int k = A.dimension(1);
  int n = B.dimension(1);
  int ldc = C.stride(0);
  gemm.MatMul(
      transa, transb,
      m, n, k,
      A.data, lda,
      B.data, ldb,
      C.data, ldc);
}

void CpuOperators::Impl::BMM_Float32(
    SubtensorCf A,
    SubtensorCf B,
    Subtensorf C) {
  CHECK(A.rank() == C.rank());
  CHECK(A.rank() >= B.rank());

  CHECK(A.dimension(0) == C.dimension(0));
  if (A.rank() == B.rank() && B.rank() == 2) {
    GEMM_Float32(A, B, C);
  } else if (A.rank() > B.rank()) {
    // broadcast B
    for (int i = 0; i < A.dimension(0); ++i) {
      SubtensorCf As = A.subtensor(i);
      Subtensorf Cs = C.subtensor(i);
      BMM_Float32(As, B, Cs);
    }
  } else if (A.rank() == B.rank()) {
    // batch dim
    CHECK(A.dimension(0) == B.dimension(0));
    for (int i = 0; i < A.dimension(0); ++i) {
      SubtensorCf As = A.subtensor(i);
      SubtensorCf Bs = B.subtensor(i);
      Subtensorf Cs = C.subtensor(i);
      BMM_Float32(As, Bs, Cs);
    }
  } else {
    NOT_IMPL();
  }
}

Tensor CpuOperators::Impl::TensorLike(SubtensorCf input) {
  std::vector<int> shape;
  for (const Size::Elem &s : input.shape) {
    shape.push_back(s.shape);
  }

  Tensor tensor;
  tensor.size_ = Size(util::MakeConstSpan(shape));

  // data
  int64_t numel = input.numel();
  tensor.data_ = std::make_shared<TensorData>(numel, TypeID<float>());
  tensor.data_ptr_ = tensor.data_->data();

  return tensor;
}

void CpuOperators::Impl::Print1D_Float32(SubtensorCf A) {
  CHECK(A.rank() == 1);

  printf("[");
  for (int i = 0; i < A.dimension(0); ++i) {
    float elem = A.elem(i);
    if (std::abs(elem) > 100 || std::abs(elem) < 0.01) {
      printf("%.4e", elem);
    } else {
      printf("%.4f", elem);
    }

    if (A.dimension(0) > kPrintEdgeItems * 2 && i == kPrintEdgeItems - 1) {
      printf(" ... ");
      i += A.dimension(0) - kPrintEdgeItems * 2;
    } else if (i < A.dimension(0) - 1) {
      printf(", ");
    }
  }
  printf("]");
}

void CpuOperators::Impl::PrintND_Float32(SubtensorCf A, int pad_space) {
  CHECK(A.rank() >= 2);

  printf("[");
  for (int i = 0; i < A.dimension(0); ++i) {
    if (i > 0) {
      for (int j = 0; j < pad_space + 1; ++j) printf(" ");
    }
    if (A.rank() == 2) {
      Print1D_Float32(A.subtensor(i));
    } else {
      PrintND_Float32(A.subtensor(i), pad_space + 1);
    }
    
    
    if (i < A.dimension(0) - 1) printf(",\n"); 
    if (A.dimension(0) > kPrintEdgeItems * 2 && i == kPrintEdgeItems - 1) {
      for (int j = 0; j < pad_space + 1; ++j) printf(" ");
      printf("...\n");
      i += A.dimension(0) - kPrintEdgeItems * 2;
    }
  }
  printf("]");
}

void CpuOperators::Impl::Print_Float32(SubtensorCf tensor) {
  int rank = tensor.rank();

  printf("tensor(");
  switch (rank) {
    case 1:
      Print1D_Float32(tensor);
      break;
    default:
      PrintND_Float32(tensor, 7);
      break;
  }
  puts(")");
}

Tensor CpuOperators::Impl::Add_Float32(SubtensorCf A, SubtensorCf B) {
  Tensor C = TensorLike(A);
  Subtensorf Cs = MakeSubTensor<float>(C);

  BinaryOp<float> add_op = [](SubtensorCf A, SubtensorCf B, Subtensorf C) {
    for (int i = 0; i < A.dimension(0); ++i) {
      C.elem(i) = A.elem(i) + B.elem(i);
    }
  };
  ApplyBinaryOperator<float>(A, B, Cs, add_op);
  return C;
}

Tensor CpuOperators::Impl::Softmax_Float32(SubtensorCf A) {
  Tensor C = TensorLike(A);
  Subtensorf Cs = MakeSubTensor<float>(C);

  auto softmax_op = [](SubtensorCf A, Subtensorf C) {
    double sum = 0;
    for (int i = 0; i < A.dimension(0); ++i) {
      float va = A.elem(i);
      sum += std::exp(va);
    }

    double logsum = std::log(sum);
    for (int i = 0; i < A.dimension(0); ++i) {
      float va = A.elem(i);;
      float &vc = C.elem(i);
      vc = static_cast<float>(std::exp(va - logsum));
    }
  };

  ApplyUnary1DTensorOp<float>(A, Cs, softmax_op);
  return C;
}

bool CpuOperators::Impl::AllClose_Float32(
    SubtensorCf A,
    SubtensorCf B,
    float rtol,
    float atol) {
  bool all_close = true;
  BinaryOpNR<float> closure = [&all_close, rtol, atol](
      SubtensorCf A,
      SubtensorCf B) {
    for (int i = 0; i < A.dimension(0); ++i) {
      float va = A.elem(i);
      float vb = B.elem(i);
      if (fabs(va - vb) > atol + rtol * fabs(vb)) {
        all_close = false;
      }
    }
  };

  ForEach<float>(A, B, closure);
  
  return all_close;
}

Tensor CpuOperators::Impl::Mul_Float32(SubtensorCf A, float k) {
  Tensor C = TensorLike(A);
  Subtensorf Cs = MakeSubTensor<float>(C);

  UnaryOp<float> unary_op = [k](SubtensorCf A, Subtensorf C) {
    for (int i = 0; i < A.dimension(0); ++i) {
      C.elem(i) = k * A.elem(i);
    }
  };

  ApplyUnary1DTensorOp<float>(A, Cs, unary_op);
  return C;
}

void CpuOperators::Impl::Copy_Float32(SubtensorCf src, Subtensorf tgt) {
  UnaryOp<float> copy_op = [](SubtensorCf src, Subtensorf tgt) {
    for (int i = 0; i < src.dimension(0); ++i) {
      tgt.elem(i) = src.elem(i);
    }
  };
  ApplyUnary1DTensorOp<float>(src, tgt, copy_op);
}

Tensor CpuOperators::Impl::Lookup_Float32(
    SubtensorCf table,
    SubtensorCl indices) {
  CHECK(table.rank() == 2 && indices.rank() == 2);

  int batch_size = indices.dimension(0);
  int seq_len = indices.dimension(1);
  int d_model = table.dimension(1);
  Tensor output = Tensor_(
      util::MakeConstSpan({batch_size, seq_len, d_model}),
      DType::kFloat);
  Subtensorf emb = MakeSubTensor<float>(output);

  for (int batch = 0; batch < batch_size; ++batch) {
    SubtensorCl indices_b = indices.subtensor(batch);
    Subtensorf emb_b = emb.subtensor(batch);
    for (int l = 0; l < seq_len; ++l) {
      int64_t index = indices_b.elem(l);
      CHECK(index < table.dimension(0)) << "indices out of range";

      SubtensorCf emb_src = table.subtensor(static_cast<int>(index));
      Subtensorf emb_tgt = emb_b.subtensor(l);
      Copy_Float32(emb_src, emb_tgt);
    }
  }

  return output;
}

Tensor CpuOperators::Impl::LayerNorm_Float32(
    SubtensorCf input,
    SubtensorCf weight,
    SubtensorCf bias,
    float eps) {
  CHECK(weight.rank() == bias.rank() && weight.rank() == 1);
  CHECK(weight.dimension(0) == bias.dimension(0));
  CHECK(input.dimension(input.rank() - 1) == weight.dimension(0));

  UnaryOp<float> closure = [&weight, &bias, eps](SubtensorCf A, Subtensorf C) {
    // mean
    double sum = 0.0f;
    for (int i = 0; i < A.dimension(0); ++i) {
      sum += A.elem(i);
    }
    double mean = sum / A.dimension(0);
    
    // var (unbiased)
    sum = 0.0;
    for (int i = 0; i < A.dimension(0); ++i) {
      double d = A.elem(i) - mean;
      sum += d * d;
    }
    double var = sum / A.dimension(0);
    double sd = sqrt(var + eps);

    // compute layer-norm
    for (int i = 0; i < A.dimension(0); ++i) {
      float elem = static_cast<float>((A.elem(i) - mean) / sd); 
      C.elem(i) = elem * weight.elem(i) + bias.elem(i);
    }
  };

  Tensor C = TensorLike(input);
  Subtensorf Cs = MakeSubTensor<float>(C);
  ApplyUnary1DTensorOp(input, Cs, closure);

  return C;
}

// ---------------------------------------------------------------------------+
// class CpuOperators                                                         |
// ---------------------------------------------------------------------------+

std::unique_ptr<Operators> CpuOperators::Create() {
  return std::unique_ptr<Operators>{new CpuOperators()};
}

Tensor CpuOperators::Tensor_(std::initializer_list<int> shape, DType dtype) {
  return impl_->Tensor_(util::MakeConstSpan(shape), dtype);
}

Tensor CpuOperators::TensorLike(const Tensor &input) {
  std::vector<Tensor::ShapeType> shape_vec;
  for (const Size::Elem &elem : input.size_.data_) {
    shape_vec.push_back(elem.shape);
  }

  return impl_->Tensor_(util::MakeConstSpan(shape_vec), input.dtype());
}

Tensor CpuOperators::Rand(std::initializer_list<int> shape, DType dtype) {
  Tensor tensor = Tensor_(shape, dtype);
  switch (dtype) {
    case DType::kFloat:
      impl_->Rand_Float32(&tensor);
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
      impl_->Zeros_Float32(impl_->MakeSubTensor<float>(tensor));
      break;
    default:
      CHECK(false) << "unsupported dtype for Zeros";
  }

  return tensor;
}

Tensor CpuOperators::MatMul(const Tensor &A, const Tensor &B) {
  switch (A.dtype()) {
    case DType::kFloat:
      return impl_->MatMul_Float32(
          impl_->MakeConstSubTensor<float>(A),
          impl_->MakeConstSubTensor<float>(B));
      break;
    default:
      CHECK(false) << "unsupported dtype for MatMul";
      return Tensor();
  }
}

void CpuOperators::Print(const Tensor &tensor) {
  switch (tensor.dtype()) {
    case DType::kFloat:
      impl_->Print_Float32(impl_->MakeConstSubTensor<float>(tensor));
      break;
    default:
      CHECK(false) << "unsupported dtype for Print";
  }
}

Tensor CpuOperators::Add(const Tensor &input, const Tensor &other) {
  switch (input.dtype()) {
    case DType::kFloat:
      return impl_->Add_Float32(
          impl_->MakeConstSubTensor<float>(input),
          impl_->MakeConstSubTensor<float>(other));
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CpuOperators::Softmax(const Tensor &input) {
  switch (input.dtype()) {
    case DType::kFloat:
      return impl_->Softmax_Float32(impl_->MakeConstSubTensor<float>(input));
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

bool CpuOperators::AllClose(const Tensor &A, const Tensor &B) {
  if (A.dtype() != B.dtype()) {
    return false;
  }

  switch (A.dtype()) {
    case DType::kFloat:
      return impl_->AllClose_Float32(
          impl_->MakeConstSubTensor<float>(A),
          impl_->MakeConstSubTensor<float>(B),
          1e-6f,
          1e-3f);
      break;
    default:
      NOT_IMPL();
  }

  return false;
}

Tensor CpuOperators::Mul(const Tensor &A, float k) {
  switch (A.dtype()) {
    case DType::kFloat:
      return impl_->Mul_Float32(impl_->MakeConstSubTensor<float>(A), k);
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CpuOperators::Contiguous(const Tensor &input) {
  if (input.is_contiguous()) {
    return input;
  } else {
    Tensor C = TensorLike(input);
    switch (input.dtype()) {
      case DType::kFloat:
        impl_->Copy_Float32(
            impl_->MakeConstSubTensor<float>(input),
            impl_->MakeSubTensor<float>(C));
        break;
      default:
        NOT_IMPL();
    }

    return C;
  }
}

Tensor CpuOperators::Lookup(const Tensor &table, const Tensor &indices) {
  switch (table.dtype()) {
    case DType::kFloat:
      return impl_->Lookup_Float32(
          impl_->MakeConstSubTensor<float>(table),
          impl_->MakeConstSubTensor<LongType>(indices));
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CpuOperators::LayerNorm(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    float eps) {
  CHECK(input.dtype() == weight.dtype() && input.dtype() == bias.dtype());

  switch (input.dtype()) {
    case DType::kFloat:
      return impl_->LayerNorm_Float32(
          impl_->MakeConstSubTensor<float>(input),
          impl_->MakeConstSubTensor<float>(weight),
          impl_->MakeConstSubTensor<float>(bias),
          eps);
    default:
      NOT_IMPL();
  }

  return Tensor();
}

}  // namespace nn
}  // namespace llama
