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

// -- class CpuOperators::Impl -------------------------------------------------

// the CPU implementation of Operators
class CPUOperators::Impl {
 public:
  typedef TensorShape::Elem Shape;

  // sub-tensor. Stores the shape and data pointer to a sub region of the 
  // original Tensor. It's faster than Tensor when passing as parameters.
  template<typename T>
  struct SubTensor;
  typedef SubTensor<float> Subtensorf;
  typedef SubTensor<const LongType> SubtensorCl;
  typedef SubTensor<const float> SubtensorCf;

  // type for lambda expression with 1D tensor as input. NR means no return.
  template<typename T>
  using UnaryOp = std::function<void(SubTensor<const T> A, SubTensor<T> C)>;
  template<typename T>
  using BinaryOp = std::function<void(SubTensor<const T> A, SubTensor<const T> B, SubTensor<T> C)>;
  template<typename T>
  using BinaryOpNR = std::function<void(SubTensor<const T> A, SubTensor<const T> B)>;


  // apply elementwise binary operator to tensor `A` and tensor `B`, then store
  // result to tensor `C`
  //     C_i = binary_op(A_i, B_i) , here A_i is 1D tensor
  // broadcast tensor `B` if necessary
  template<typename T>
  void ApplyBinaryOperator(
      SubTensor<const T> A, SubTensor<const T> B, SubTensor<T> C, BinaryOp<T> binary_op);

  // apply unary 1D tensor operator on the last dimension of `A` and save
  // result to `C`. `C` should have the same shape as A.
  //   C_i = tensor1d_op(A_i) , here A_i is 1D tensor
  template<typename T>
  void ApplyUnary1DTensorOp(SubTensor<const T> A, SubTensor<T> C, UnaryOp<T> unary_op);

  // call closure on all elements in tensor A and tensor B. Shape of A should
  // equal to B.
  //   closure(A_i, B_i) , here A_i and B_i are 1D tensor.
  template<typename T>
  void ForEach(SubTensor<const T> A, SubTensor<const T> B, BinaryOpNR<T> binary_op);

  // Get sub-tensor from Tensor.
  template<typename T>
  SubTensor<T> makeSubtensor(Tensor &tensor); 
  template<typename T>
  SubTensor<const T> makeConstSubtensor(const Tensor &tensor); 

  Tensor createTensor(util::Span<const int> shape, DType dtype);
  Tensor createTensorLike(SubtensorCf A);

  Tensor matmulFp32(SubtensorCf A, SubtensorCf B);
  Tensor gemvFp32(SubtensorCf A, SubtensorCf B);
  Tensor bmmFp32(SubtensorCf A, SubtensorCf B);
  Tensor bmvFp32(SubtensorCf A, SubtensorCf B);
  Tensor gemmFp32(SubtensorCf A, SubtensorCf B);

  Tensor mulFp32(SubtensorCf A, float k);
  Tensor addFp32(SubtensorCf A, SubtensorCf B);
  Tensor softmaxFp32(SubtensorCf A);
  Tensor geluFp32(SubtensorCf A);

  void randFp32(Tensor *tensor);
  void zerosFp32(Subtensorf tensor);
  void print1DFp32(SubtensorCf A);
  void printNDFp32(SubtensorCf A, int pad_space);
  void printFp32(SubtensorCf tensor);
  
  bool allCloseFp32(SubtensorCf A, SubtensorCf B, float rtol, float atol);

  void catFp32(SubtensorCf A, SubtensorCf B, int dim, Subtensorf C);

  // Copy data from src to tgt. tgt and src should have the same dtype and
  // shape
  void copyFp32(SubtensorCf src, Subtensorf tgt);

  Tensor lookupFp32(SubtensorCf table, SubtensorCl indices);
  Tensor layerNormFp32(
      SubtensorCf input,
      SubtensorCf weight,
      SubtensorCf bias,
      float eps);
  Tensor causalMaskFp32(int max_len);

 private:
  GEMM _gemm;

  // generate GEMMArgs for A * B -> C.
  GEMMArgs generateGemmArgs(SubtensorCf A, SubtensorCf B, Subtensorf C);

  // generate GEMVArgs for A * B -> C.
  GEMVArgs generateGemvArgs(SubtensorCf A, SubtensorCf B, Subtensorf C);
};

// -- class CPUOperators::Impl::SubTensor ----------

// sub-tensor. Stores the shape and data pointer to a sub region of the original
// Tensor. It's faster than Tensor when passing as parameters.
template<typename T>
struct CPUOperators::Impl::SubTensor {
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
inline auto CPUOperators::Impl::makeSubtensor(Tensor &tensor) -> SubTensor<T> {
  return SubTensor<T>{util::makeConstSpan(tensor._shape._data), tensor.getData<T>()};
} 

template<typename T>
inline auto CPUOperators::Impl::makeConstSubtensor(const Tensor &tensor) -> SubTensor<const T> {
  return SubTensor<const T>{util::makeConstSpan(tensor._shape._data), tensor.getData<T>()};
} 

// ---------------------------------------------------------------------------+
// CpuOperators::Impl                                                         |
// ---------------------------------------------------------------------------+

template<typename T>
void CPUOperators::Impl::ApplyBinaryOperator(
    SubTensor<const T> A, SubTensor<const T> B, SubTensor<T> C, BinaryOp<T> binary_op) {
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
void CPUOperators::Impl::ApplyUnary1DTensorOp(
    SubTensor<const T> A, SubTensor<T> C, UnaryOp<T> unary_op) {
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
void CPUOperators::Impl::ForEach(
    SubTensor<const T> A, SubTensor<const T> B, BinaryOpNR<T> closure) {
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

Tensor CPUOperators::Impl::createTensor(util::Span<const int> shape, DType dtype) {
  Tensor tensor;

  tensor._shape = TensorShape(util::makeConstSpan(shape));
  int64_t numel = tensor._shape.getNumEl();

  tensor._data = std::make_shared<TensorData>(numel, dtype);
  tensor._dataPtr = tensor._data->getData();

  return tensor;
}

void CPUOperators::Impl::randFp32(Tensor *tensor) {
  float *data = tensor->getData<float>();
  int64_t numel = tensor->getNumEl();

  float randmax = RAND_MAX;
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = ::rand() / randmax - 0.5f;
  }
}

void CPUOperators::Impl::zerosFp32(Subtensorf tensor) {
  // make sure tensor is contiguous.
  CHECK(tensor.numel() == tensor.stride(0) * tensor.dimension(0));

  float *data = tensor.data;
  int64_t numel = tensor.numel();

  for (int64_t i = 0; i < numel; ++i) {
    data[i] = 0.0f;
  }
}

GEMMArgs CPUOperators::Impl::generateGemmArgs(SubtensorCf A, SubtensorCf B, Subtensorf C) {
  CHECK(A.rank() == B.rank() && A.rank() == C.rank());
  CHECK(A.rank() == 2);
  CHECK(A.dimension(0) == C.dimension(0));
  CHECK(A.dimension(1) == B.dimension(0));
  CHECK(B.dimension(1) == C.dimension(1));
  
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

  GEMMArgs gemmArgs;
  gemmArgs.A = A.data;
  gemmArgs.B = B.data;
  gemmArgs.C = C.data;
  gemmArgs.K = k;
  gemmArgs.lda = lda;
  gemmArgs.ldb = ldb;
  gemmArgs.ldc = ldc;
  gemmArgs.M = m;
  gemmArgs.N = n;
  gemmArgs.TransA = transa;
  gemmArgs.TransB = transb;

  return gemmArgs;
}


GEMVArgs CPUOperators::Impl::generateGemvArgs(SubtensorCf A, SubtensorCf B, Subtensorf C) {
  CHECK(A.rank() == 2 && B.rank() == 1 && A.dimension(1) == B.dimension(0));
  CHECK(C.rank() == 1 && C.dimension(0) == A.dimension(0));

  bool transA;
  int lda, m, n;
  if (A.stride(1) == 1) {
    transA = false;
    lda = A.stride(0);
    m = A.dimension(0);
    n = A.dimension(1);
  } else if (A.stride(0) == 1) {
    transA = true;
    lda = A.stride(1);
    m = A.dimension(1);
    n = A.dimension(0);
  } else {
    NOT_IMPL();
  }

  // when transA == true: len(x, y) = (m, n)
  // when transA == false: len(x, y) = (n, m)
  // x is B
  // y is C
  GEMVArgs gemvArgs;
  gemvArgs.A = A.data;
  gemvArgs.lda = lda;
  gemvArgs.M = m;
  gemvArgs.N = n;
  gemvArgs.TransA = transA;
  gemvArgs.x = B.data;
  gemvArgs.y = C.data;

  return gemvArgs;
}

Tensor CPUOperators::Impl::gemmFp32(SubtensorCf A, SubtensorCf B) {
  CHECK(A.rank() == B.rank() && A.rank() == 2);

  Tensor C = createTensor(util::makeConstSpan({A.dimension(0), B.dimension(1)}), DType::kFloat);
  Subtensorf Cs = makeSubtensor<float>(C);
  zerosFp32(Cs);

  GEMMArgs gemmArgs = generateGemmArgs(A, B, Cs);
  _gemm.sgemm(gemmArgs);

  return C;
}

Tensor CPUOperators::Impl::gemvFp32(SubtensorCf A, SubtensorCf B) {
  CHECK(A.rank() == 2 && B.rank() == 1);

  Tensor C = createTensor({A.dimension(0)}, DType::kFloat);
  Subtensorf Cs = makeSubtensor<float>(C);
  zerosFp32(Cs);

  GEMVArgs gemvArgs = generateGemvArgs(A, B, Cs);
  _gemm.sgemv(gemvArgs);
  return C;
}

Tensor CPUOperators::Impl::bmmFp32(SubtensorCf A, SubtensorCf B) {
  CHECK(A.rank() >= B.rank() && A.rank() > 2 && A.rank() <= 4 && B.rank() >= 2);
  std::vector<int> shape;

  // broadcast B
  int broadcastDims = A.rank() - B.rank();
  for (int i = 0; i < broadcastDims; ++i) {
    shape.push_back(A.dimension(i));
  }

  // batch dim: B.shape(i) == A.shape(broadcastDims + i)
  int batchDims = B.rank() - 2;
  for (int i = 0; i < batchDims; ++i) {
    CHECK(A.dimension(broadcastDims + i) == B.dimension(i));
    shape.push_back(B.dimension(i));
  }

  // GEMM dims (A.shape(-2), B.shape(-1))
  shape.push_back(A.dimension(broadcastDims + batchDims));
  shape.push_back(B.dimension(batchDims + 1));

  Tensor tensorC = createTensor(shape, DType::kFloat);
  Subtensorf C = makeSubtensor<float>(tensorC);
  zerosFp32(C);

  std::vector<GEMMArgs> batchArgs;
  if (A.rank() == 3) {
    bool broadcastB = B.rank() == 2;
    for (int i = 0; i < A.dimension(0); ++i) {
      SubtensorCf Bs = broadcastB ? B : B.subtensor(i);
      batchArgs.push_back(generateGemmArgs(A.subtensor(i), Bs, C.subtensor(i)));
    }
  } else if (A.rank() == 4) {
    bool broadcastB = B.rank() < 4;
    for (int i = 0; i < A.dimension(0); ++i) {
      SubtensorCf As = A.subtensor(i);
      SubtensorCf Bs = broadcastB ? B : B.subtensor(i);
      Subtensorf Cs = C.subtensor(i);
      bool broadcastBs = B.rank() < 3;
      for (int j = 0; j < As.dimension(0); ++j) {
        SubtensorCf Bs0 = broadcastBs ? Bs : Bs.subtensor(j);
        batchArgs.push_back(generateGemmArgs(As.subtensor(j), Bs0, Cs.subtensor(j)));
      }
    }
  }

  _gemm.sgemmBatch(batchArgs);
  return tensorC;
}

Tensor CPUOperators::Impl::bmvFp32(SubtensorCf A, SubtensorCf B) {
  CHECK(A.rank() - B.rank() == 1 && B.rank() >= 2);

  std::vector<int> shape;
  int batchDims = B.rank() - 1;
  for (int d = 0; d < batchDims; ++d) {
    int dimA = A.dimension(d);
    int dimB = B.dimension(d);

    CHECK(dimA == 1 || dimB == 1 || dimA == dimB);
    shape.push_back(std::max(dimA, dimB));
  }

  shape.push_back(A.dimension(batchDims));
  Tensor tensorC = createTensor(shape, DType::kFloat);
  Subtensorf C = makeSubtensor<float>(tensorC);

  std::vector<GEMVArgs> batchArgs;
  if (A.rank() == 3) {
    for (int i = 0; i < shape[0]; ++i) {
      SubtensorCf As = A.dimension(0) == 1 ? A.subtensor(0) : A.subtensor(i);
      SubtensorCf Bs = B.dimension(0) == 1 ? B.subtensor(0) : B.subtensor(i);

      batchArgs.push_back(generateGemvArgs(As, Bs, C.subtensor(i)));
    }
  } else if (A.rank() == 4) {
    for (int i = 0; i < shape[0]; ++i) {
      SubtensorCf As = A.dimension(0) == 1 ? A.subtensor(0) : A.subtensor(i);
      SubtensorCf Bs = B.dimension(0) == 1 ? B.subtensor(0) : B.subtensor(i);
      Subtensorf Cs = C.subtensor(i);
      for (int j = 0; j < shape[1]; ++j) {
        SubtensorCf As0 = As.dimension(0) == 1 ? As.subtensor(0) : As.subtensor(i);
        SubtensorCf Bs0 = Bs.dimension(0) == 1 ? Bs.subtensor(0) : Bs.subtensor(i);
        batchArgs.push_back(generateGemvArgs(As0, Bs0, Cs.subtensor(i)));
      }
    }
  }

  _gemm.sgemvBatch(batchArgs);
  return tensorC;
}

Tensor CPUOperators::Impl::createTensorLike(SubtensorCf input) {
  std::vector<int> shape;
  for (const Shape &s : input.shape) {
    shape.push_back(s.shape);
  }

  Tensor tensor;
  tensor._shape = TensorShape(util::makeConstSpan(shape));

  // data
  int64_t numel = input.numel();
  tensor._data = std::make_shared<TensorData>(numel, getTypeID<float>());
  tensor._dataPtr = tensor._data->getData();

  return tensor;
}

void CPUOperators::Impl::print1DFp32(SubtensorCf A) {
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

void CPUOperators::Impl::printNDFp32(SubtensorCf A, int pad_space) {
  CHECK(A.rank() >= 2);

  printf("[");
  for (int i = 0; i < A.dimension(0); ++i) {
    if (i > 0) {
      for (int j = 0; j < pad_space + 1; ++j) printf(" ");
    }
    if (A.rank() == 2) {
      print1DFp32(A.subtensor(i));
    } else {
      printNDFp32(A.subtensor(i), pad_space + 1);
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

void CPUOperators::Impl::printFp32(SubtensorCf tensor) {
  int rank = tensor.rank();

  printf("tensor(");
  switch (rank) {
    case 1:
      print1DFp32(tensor);
      break;
    default:
      printNDFp32(tensor, 7);
      break;
  }
  printf(", shape=(");
  for (int d = 0; d < tensor.rank(); ++d) {
    if (d) printf(", ");
    printf("%d", tensor.dimension(d));
  }
  puts("))");
}

Tensor CPUOperators::Impl::addFp32(SubtensorCf A, SubtensorCf B) {
  Tensor C = createTensorLike(A);
  Subtensorf Cs = makeSubtensor<float>(C);

  BinaryOp<float> add_op = [](SubtensorCf A, SubtensorCf B, Subtensorf C) {
    for (int i = 0; i < A.dimension(0); ++i) {
      C.elem(i) = A.elem(i) + B.elem(i);
    }
  };
  ApplyBinaryOperator<float>(A, B, Cs, add_op);
  return C;
}

Tensor CPUOperators::Impl::softmaxFp32(SubtensorCf A) {
  Tensor C = createTensorLike(A);
  Subtensorf Cs = makeSubtensor<float>(C);

  auto softmax_op = [](SubtensorCf A, Subtensorf C) {
    double sum = 0;
    for (int i = 0; i < A.dimension(0); ++i) {
      float va = A.elem(i);
      sum += std::exp(va);
    }

    double logsum = std::log(sum);
    for (int i = 0; i < A.dimension(0); ++i) {
      float va = A.elem(i);
      float &vc = C.elem(i);
      vc = static_cast<float>(std::exp(va - logsum));
    }
  };

  ApplyUnary1DTensorOp<float>(A, Cs, softmax_op);
  return C;
}

Tensor CPUOperators::Impl::geluFp32(SubtensorCf A) {
  Tensor C = createTensorLike(A);
  Subtensorf Cs = makeSubtensor<float>(C);

  auto gelu_op = [](SubtensorCf A, Subtensorf C) {
    for (int i = 0; i < A.dimension(0); ++i) {
      float x = A.elem(i);

      double x3 = pow(x, 3.0);
      double c = 0.5 * x * (1 + tanh(sqrt(2.0 / kPi) * (x + 0.044715 * x3)));
      C.elem(i) = static_cast<float>(c);
    }
  };

  ApplyUnary1DTensorOp<float>(A, Cs, gelu_op);
  return C;
}

bool CPUOperators::Impl::allCloseFp32(SubtensorCf A, SubtensorCf B, float rtol, float atol) {
  bool all_close = true;
  BinaryOpNR<float> closure = [&all_close, rtol, atol](SubtensorCf A, SubtensorCf B) {
    for (int i = 0; i < A.dimension(0); ++i) {
      float va = A.elem(i);
      float vb = B.elem(i);
      if (!(std::isfinite(va) && std::isfinite(vb))) {
        all_close = false;
      }
      if (fabs(va - vb) > atol + rtol * fabs(vb)) {
        all_close = false;
      }
    }
  };

  ForEach<float>(A, B, closure);
  
  return all_close;
}

Tensor CPUOperators::Impl::mulFp32(SubtensorCf A, float k) {
  Tensor C = createTensorLike(A);
  Subtensorf Cs = makeSubtensor<float>(C);

  UnaryOp<float> unary_op = [k](SubtensorCf A, Subtensorf C) {
    for (int i = 0; i < A.dimension(0); ++i) {
      C.elem(i) = k * A.elem(i);
    }
  };

  ApplyUnary1DTensorOp<float>(A, Cs, unary_op);
  return C;
}

void CPUOperators::Impl::copyFp32(SubtensorCf src, Subtensorf tgt) {
  UnaryOp<float> copy_op = [](SubtensorCf src, Subtensorf tgt) {
    for (int i = 0; i < src.dimension(0); ++i) {
      tgt.elem(i) = src.elem(i);
    }
  };
  ApplyUnary1DTensorOp<float>(src, tgt, copy_op);
}

Tensor CPUOperators::Impl::lookupFp32(SubtensorCf table, SubtensorCl indices) {
  CHECK(table.rank() == 2 && indices.rank() == 2);

  int batch_size = indices.dimension(0);
  int seq_len = indices.dimension(1);
  int d_model = table.dimension(1);
  Tensor output = createTensor(util::makeConstSpan({batch_size, seq_len, d_model}), DType::kFloat);
  Subtensorf emb = makeSubtensor<float>(output);

  for (int batch = 0; batch < batch_size; ++batch) {
    SubtensorCl indices_b = indices.subtensor(batch);
    Subtensorf emb_b = emb.subtensor(batch);
    for (int l = 0; l < seq_len; ++l) {
      int64_t index = indices_b.elem(l);
      CHECK(index < table.dimension(0)) << "indices out of range";

      SubtensorCf emb_src = table.subtensor(static_cast<int>(index));
      Subtensorf emb_tgt = emb_b.subtensor(l);
      copyFp32(emb_src, emb_tgt);
    }
  }

  return output;
}

Tensor CPUOperators::Impl::layerNormFp32(
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

  Tensor C = createTensorLike(input);
  Subtensorf Cs = makeSubtensor<float>(C);
  ApplyUnary1DTensorOp(input, Cs, closure);

  return C;
}

Tensor CPUOperators::Impl::causalMaskFp32(int seq_len) {
  Tensor mask = createTensor(util::makeConstSpan({seq_len, seq_len}), DType::kFloat);
  CHECK(mask.isContiguous());

  float *data = mask.getData<float>();
  for (int i = 0; i < seq_len; ++i) {
    float *row = data + i * seq_len;
    for (int j = 0; j <= i; ++j) {
      row[j] = 0.0f;
    }
    for (int j = i + 1; j < seq_len; ++j) {
      row[j] = -std::numeric_limits<float>::infinity();
    }
  }

  return mask;
}

void CPUOperators::Impl::catFp32(SubtensorCf A, SubtensorCf B, int dim, Subtensorf C) {
  CHECK(A.rank() == B.rank() && A.rank() == C.rank());
  if (dim == 0) {
    CHECK(A.dimension(0) + B.dimension(0) == C.dimension(0));
    for (int i = 0; i < A.dimension(0); ++i) {
      copyFp32(A.subtensor(i), C.subtensor(i));
    }
    for (int i = 0; i < B.dimension(0); ++i) {
      copyFp32(B.subtensor(i), C.subtensor(i + A.dimension(0)));
    }
  } else {
    CHECK(A.dimension(0) == B.dimension(0));
    for (int i = 0; i < A.dimension(0); ++i) {
      catFp32(A.subtensor(i), B.subtensor(i), dim - 1, C.subtensor(i));
    }
  }
}

// -- class CPUOperators ----------

std::unique_ptr<Operators> CPUOperators::create() {
  std::unique_ptr<CPUOperators> F{new CPUOperators()};
  F->_impl = std::make_unique<Impl>();

  return F;
}

Tensor CPUOperators::createTensor(std::initializer_list<int> shape, DType dtype) {
  return _impl->createTensor(util::makeConstSpan(shape), dtype);
}

Tensor CPUOperators::createTensorLike(const Tensor &input) {
  std::vector<Tensor::ShapeType> shape_vec;
  for (const TensorShape::Elem &elem : input._shape._data) {
    shape_vec.push_back(elem.shape);
  }

  return _impl->createTensor(util::makeConstSpan(shape_vec), input.getDType());
}

Tensor CPUOperators::rand(std::initializer_list<int> shape, DType dtype) {
  Tensor tensor = createTensor(shape, dtype);
  switch (dtype) {
    case DType::kFloat:
      _impl->randFp32(&tensor);
      break;
    default:
      CHECK(false) << "unsupported dtype for Rand";
  }

  return tensor;
}

Tensor CPUOperators::zeros(std::initializer_list<int> shape, DType dtype) {
  Tensor tensor = createTensor(shape, dtype);
  switch (dtype) {
    case DType::kFloat:
      _impl->zerosFp32(_impl->makeSubtensor<float>(tensor));
      break;
    default:
      CHECK(false) << "unsupported dtype for Zeros";
  }

  return tensor;
}

Tensor CPUOperators::gemm(const Tensor &A, const Tensor &B) {
  switch (A.getDType()) {
    case DType::kFloat:
      return _impl->gemmFp32(
          _impl->makeConstSubtensor<float>(A), _impl->makeConstSubtensor<float>(B));
      break;
    default:
      CHECK(false) << "unsupported dtype for MatMul";
      return Tensor();
  }
}

Tensor CPUOperators::bmm(const Tensor &A, const Tensor &B) {
  switch (A.getDType()) {
    case DType::kFloat:
      return _impl->bmmFp32(
          _impl->makeConstSubtensor<float>(A), _impl->makeConstSubtensor<float>(B));
    default:
      CHECK(false) << "unsupported dtype for MatMul";
      return Tensor();
  }
}

Tensor CPUOperators::bmv(const Tensor &A, const Tensor &B) {
  switch (A.getDType()) {
    case DType::kFloat:
      return _impl->bmvFp32(
          _impl->makeConstSubtensor<float>(A), _impl->makeConstSubtensor<float>(B));
      break;
    default:
      CHECK(false) << "unsupported dtype for bmv";
      return Tensor();
  }
}

Tensor CPUOperators::gemv(const Tensor &A, const Tensor &B) {
  switch (A.getDType()) {
    case DType::kFloat:
      return _impl->gemvFp32(
          _impl->makeConstSubtensor<float>(A), _impl->makeConstSubtensor<float>(B));
      break;
    default:
      CHECK(false) << "unsupported dtype for gemv";
      return Tensor();
  }
}

void CPUOperators::print(const Tensor &tensor) {
  switch (tensor.getDType()) {
    case DType::kFloat:
      _impl->printFp32(_impl->makeConstSubtensor<float>(tensor));
      break;
    default:
      CHECK(false) << "unsupported dtype for Print";
  }
}

Tensor CPUOperators::add(const Tensor &input, const Tensor &other) {
  switch (input.getDType()) {
    case DType::kFloat:
      return _impl->addFp32(
          _impl->makeConstSubtensor<float>(input), _impl->makeConstSubtensor<float>(other));
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::softmax(const Tensor &input) {
  switch (input.getDType()) {
    case DType::kFloat:
      return _impl->softmaxFp32(_impl->makeConstSubtensor<float>(input));
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::gelu(const Tensor &input) {
  switch (input.getDType()) {
    case DType::kFloat:
      return _impl->geluFp32(_impl->makeConstSubtensor<float>(input));
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

bool CPUOperators::allClose(const Tensor &A, const Tensor &B) {
  if (A.getDType() != B.getDType()) {
    return false;
  }

  switch (A.getDType()) {
    case DType::kFloat:
      return _impl->allCloseFp32(
          _impl->makeConstSubtensor<float>(A), _impl->makeConstSubtensor<float>(B), 1e-6f, 1e-3f);
      break;
    default:
      NOT_IMPL();
  }

  return false;
}

Tensor CPUOperators::mul(const Tensor &A, float k) {
  switch (A.getDType()) {
    case DType::kFloat:
      return _impl->mulFp32(_impl->makeConstSubtensor<float>(A), k);
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::contiguous(const Tensor &input) {
  if (input.isContiguous()) {
    return input;
  } else {
    Tensor C = createTensorLike(input);
    switch (input.getDType()) {
      case DType::kFloat:
        _impl->copyFp32(_impl->makeConstSubtensor<float>(input), _impl->makeSubtensor<float>(C));
        break;
      default:
        NOT_IMPL();
    }

    return C;
  }
}

Tensor CPUOperators::lookup(const Tensor &table, const Tensor &indices) {
  switch (table.getDType()) {
    case DType::kFloat:
      return _impl->lookupFp32(
          _impl->makeConstSubtensor<float>(table),
          _impl->makeConstSubtensor<LongType>(indices));
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::layerNorm(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    float eps) {
  CHECK(input.getDType() == weight.getDType() && input.getDType() == bias.getDType());

  switch (input.getDType()) {
    case DType::kFloat:
      return _impl->layerNormFp32(
          _impl->makeConstSubtensor<float>(input),
          _impl->makeConstSubtensor<float>(weight),
          _impl->makeConstSubtensor<float>(bias),
          eps);
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::causalMask(int max_len) {
  return _impl->causalMaskFp32(max_len);
}


Tensor CPUOperators::cat(const Tensor &A, const Tensor &B, int dim) {
  CHECK(A.getDim() == B.getDim());

  std::vector<int> shape;
  for (int d = 0; d < A.getDim(); ++d) {
    if (d == dim) {
      shape.push_back(A.getShape(d) + B.getShape(d));
    } else {
      CHECK(A.getShape(d) == B.getShape(d));
      shape.push_back(A.getShape(d));
    }
  }

  Tensor C = _impl->createTensor(util::makeConstSpan(shape), A.getDType());
  switch (A.getDType()) {
    case DType::kFloat:
      _impl->catFp32(
          _impl->makeConstSubtensor<float>(A),
          _impl->makeConstSubtensor<float>(B),
          dim,
          _impl->makeSubtensor<float>(C));
      break;
    default:
      NOT_IMPL();
  }

  return C;
}

}  // namespace nn
}  // namespace llama
