#include "flint/cpu_operators.h"

#include <stdlib.h>
#include <cmath>
#include <limits>
#include <memory>
#include "pmpack/pmpack.h"
#include "flint/nn.h"
#include "flint/operators.h"
#include "flint/tensor.h"
#include "flint/util.h"

namespace flint {

constexpr int kPrintEdgeItems = 3;

template<typename T>
void CPUOperators::getSubtensors(Subtensor<T> tensor, int subtensorDim, std::vector<T *> &l) {
  CHECK(tensor.rank() >= subtensorDim);

  if (tensor.rank() == subtensorDim) {
    l.push_back(tensor.data);
  } else {
    for (int i = 0; i < tensor.dimension(0); ++i) {
      getSubtensors(tensor.subtensor(i), subtensorDim, l);
    }
  }
}

template<typename T>
bool CPUOperators::isShapeMatch(Subtensor<T> A, Subtensor<T> B) {
  if (A.rank() != B.rank())
    return false;

  for (int d = 0; d < A.rank(); ++d) {
    if (A.dimension(d) != B.dimension(d))
      return false;
  }

  return true;
}

template<typename T>
bool CPUOperators::isShapeMatchBroadcastB(Subtensor<T> A, Subtensor<T> B) {
  if (A.rank() < B.rank())
    return false;
  
  if (A.rank() > B.rank()) {
    A.shape = A.shape.subspan(A.rank() - B.rank());
  }

  return isShapeMatch(A, B);
}

template<typename T>
CPUOperators::SubtensorList<T> CPUOperators::getVectorList(Subtensor<T> tensor) {
  std::vector<T *> l;
  getSubtensors(tensor, 1, l);

  ly::Span<const Shape> vecShape = tensor.shape.subspan(tensor.rank() - 1);
  return SubtensorList<T>(vecShape, std::move(l));
}

template<typename T>
CPUOperators::SubtensorList<T> CPUOperators::getMatrixList(Subtensor<T> tensor) {
  std::vector<T *> l;
  getSubtensors(tensor, 2, l);

  ly::Span<const Shape> mShape = tensor.shape.subspan(tensor.rank() - 2);
  return SubtensorList<T>(mShape, std::move(l));
}

template<typename T>
inline auto CPUOperators::makeSubtensor(Tensor &tensor) -> Subtensor<T> {
  return Subtensor<T>{ly::makeConstSpan(tensor._shape._data), tensor.getData<T>()};
} 

template<typename T>
inline auto CPUOperators::makeConstSubtensor(const Tensor &tensor) -> Subtensor<const T> {
  return Subtensor<const T>{ly::makeConstSpan(tensor._shape._data), tensor.getData<T>()};
} 

CPUOperators::CPUOperators() {}

Tensor CPUOperators::createTensor(ly::Span<const int> shape, DType dtype) {
  Tensor tensor;

  tensor._shape = TensorShape(ly::makeConstSpan(shape));
  int64_t numel = tensor._shape.getNumEl();

  tensor._data = TensorData::create(numel, dtype);
  tensor._dataPtr = tensor._data->getData();

  return tensor;
}

void CPUOperators::randFp32(Tensor *tensor) {
  float *data = tensor->getData<float>();
  int64_t numel = tensor->getNumEl();

  float randmax = RAND_MAX;
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = ::rand() / randmax - 0.5f;
  }
}

void CPUOperators::zerosFp32(Subtensor<float> tensor) {
  // make sure tensor is contiguous.
  CHECK(tensor.numel() == tensor.stride(0) * tensor.dimension(0));

  float *data = tensor.data;
  int64_t numel = tensor.numel();

  for (int64_t i = 0; i < numel; ++i) {
    data[i] = 0.0f;
  }
}

CPUOperators::GEMMArgs CPUOperators::generateGemmArgs(
    const Tensor &A, const Tensor &B, const Tensor &C) {
  CHECK(A.getDim() >= B.getDim() && A.getDim() == C.getDim());
  CHECK(B.getDim() >= 2);
  CHECK(A.getShape(-2) == C.getShape(-2));
  CHECK(A.getShape(-1) == B.getShape(-2));
  CHECK(B.getShape(-1) == C.getShape(-1));

  bool transA, transB;
  int lda, ldb;
  if (A.getStride(-1) == 1) {
    transA = false;
    lda = A.getStride(-2);
  } else if (A.getStride(-2) == 1) {
    transA = true;
    lda = A.getStride(-1);
  } else {
    NOT_IMPL();
  }

  if (B.getStride(-1) == 1) {
    transB = false;
    ldb = B.getStride(-2);
  } else if (B.getStride(-2) == 1) {
    transB = true;
    ldb = B.getStride(-1);
  } else {
    NOT_IMPL();
  }

  int m = A.getShape(-2);
  int k = A.getShape(-1);
  int n = B.getShape(-1);
  int ldc = C.getStride(-2);

  GEMMArgs gemmArgs;
  gemmArgs.K = k;
  gemmArgs.lda = lda;
  gemmArgs.ldb = ldb;
  gemmArgs.ldc = ldc;
  gemmArgs.M = m;
  gemmArgs.N = n;
  gemmArgs.transA = transA;
  gemmArgs.transB = transB;

  return gemmArgs;
}

Tensor CPUOperators::matmulFp32(const Tensor &A, const Tensor &B) {
  if (A.getDim() == 2 && B.getDim() == 2) {
    return gemmFp32(A, B);
  } else if (A.getDim() >= 2 && B.getDim() >= 2) {
    return bmmFp32(A, B);
  } else {
    NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::gemmFp32(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2);

  Tensor C = createTensor({A.getShape(0), B.getShape(1)}, DType::kFloat);
  Subtensor<float> Cs = makeSubtensor<float>(C);
  zerosFp32(Cs);

  GEMMArgs gemmArgs = generateGemmArgs(A, B, C);
  pmpack_sgemm(
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      A.getData<float>(),
      gemmArgs.lda,
      B.getData<float>(),
      gemmArgs.ldb,
      Cs.data,
      gemmArgs.ldc);

  return C;
}

Tensor CPUOperators::gemmFp32QInt4Fp32(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2);

  Tensor C = createTensor({A.getShape(0), B.getShape(1)}, DType::kFloat);
  Subtensor<float> Cs = makeSubtensor<float>(C);
  zerosFp32(Cs);

  CHECK(B.getDType() == DType::kQInt4Fp32);
  const TensorData *dataObjB = B.getDataObject();

  GEMMArgs gemmArgs = generateGemmArgs(A, B, C);
  pmpack_gemm_fp32qint4fp32(
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      A.getData<float>(),
      gemmArgs.lda,
      dataObjB->getData(),
      dataObjB->getScaleData<float>(),
      dataObjB->getGroupSize(),
      C.getData<float>(),
      gemmArgs.ldc);

  return C;
}

std::vector<int> CPUOperators::getBmmOutputShape(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() >= B.getDim());
  CHECK(A.getDim() > 2 && A.getDim() <= 4 && B.getDim() >= 2);
  std::vector<int> shape;

  // broadcast B
  int broadcastDims = A.getDim() - B.getDim();
  for (int i = 0; i < broadcastDims; ++i) {
    shape.push_back(A.getShape(i));
  }

  // batch dim: B.shape(i) == A.shape(broadcastDims + i)
  int batchDims = B.getDim() - 2;
  for (int i = 0; i < batchDims; ++i) {
    CHECK(A.getShape(broadcastDims + i) == B.getShape(i));
    shape.push_back(B.getShape(i));
  }

  shape.push_back(A.getShape(-2));
  shape.push_back(B.getShape(-1));

  return shape;
}

Tensor CPUOperators::bmmFp32(const Tensor &A, const Tensor &B) {
  std::vector<int> shapeC = getBmmOutputShape(A, B);

  Tensor tensorC = createTensor(shapeC, DType::kFloat);
  Subtensor<float> C = makeSubtensor<float>(tensorC);
  zerosFp32(C);

  SubtensorList<const float> mAs = getMatrixList(makeConstSubtensor<float>(A));
  SubtensorList<const float> mBs = getMatrixList(makeConstSubtensor<float>(B));
  SubtensorList<float> mCs = getMatrixList(C);

  GEMMArgs gemmArgs = generateGemmArgs(A, B, tensorC);

  // broadcast B.
  CHECK(mAs.getSize() == mCs.getSize());
  CHECK(mAs.getSize() % mBs.getSize() == 0);
  int nb = mAs.getSize() / mBs.getSize();
  std::vector<const float*> batchB = repeat(ly::makeConstSpan(mBs.getDataPtrList()), nb);

  pmpack_sgemm_batch(
      mAs.getSize(),
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      mAs.getDataPtrList().data(),
      gemmArgs.lda,
      batchB.data(),
      gemmArgs.ldb,
      mCs.getDataPtrList().data(),
      gemmArgs.ldc);

  return tensorC;
}

Tensor CPUOperators::bmmFp32QInt4Fp32(const Tensor &A, const Tensor &B) {
  // currently only support 1 QInt4 matrix in BMM.
  CHECK(B.getDim() == 2);

  std::vector<int> shapeC = getBmmOutputShape(A, B);

  Tensor tensorC = createTensor(shapeC, DType::kFloat);
  Subtensor<float> C = makeSubtensor<float>(tensorC);
  zerosFp32(C);

  SubtensorList<const float> mAs = getMatrixList(makeConstSubtensor<float>(A));
  SubtensorList<float> mCs = getMatrixList(C);

  GEMMArgs gemmArgs = generateGemmArgs(A, B, tensorC);

  CHECK(B.getDType() == DType::kQInt4Fp32);
  const TensorData* dataObjB = B.getDataObject();

  // broadcast B (batch size for QInt4Fp32 tensor B is always 1).
  CHECK(mAs.getSize() == mCs.getSize());

  const void* dataB = dataObjB->getData();
  const float* scaleB = dataObjB->getScaleData<float>();
  int batchSize = mAs.getSize();
  std::vector<const void*> batchB = repeat(ly::makeConstSpan(&dataB, 1), batchSize);
  std::vector<const float*> batchScaleB = repeat(ly::makeConstSpan(&scaleB, 1), batchSize);
  CHECK(mAs.getSize() == batchB.size());

  pmpack_gemm_fp32qint4fp32_batch(
      mAs.getSize(),
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      mAs.getDataPtrList().data(),
      gemmArgs.lda,
      batchB.data(),
      batchScaleB.data(),
      dataObjB->getGroupSize(),
      mCs.getDataPtrList().data(),
      gemmArgs.ldc);

  return tensorC;
}

Tensor CPUOperators::createTensorLike(Subtensor<const float> input) {
  std::vector<int> shape;
  for (const Shape &s : input.shape) {
    shape.push_back(s.shape);
  }

  Tensor tensor;
  tensor._shape = TensorShape(ly::makeConstSpan(shape));

  // data
  int64_t numel = input.numel();
  tensor._data = TensorData::create(numel, getTypeID<float>());
  tensor._dataPtr = tensor._data->getData();

  return tensor;
}

void CPUOperators::print1DFp32(Subtensor<const float> A) {
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

void CPUOperators::printNDFp32(Subtensor<const float> A, int pad_space) {
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

void CPUOperators::printFp32(Subtensor<const float> tensor) {
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

Tensor CPUOperators::addFp32(Subtensor<const float> A, Subtensor<const float> B) {
  CHECK(isShapeMatchBroadcastB(A, B));

  Tensor C = createTensorLike(A);
  Subtensor<float> Cs = makeSubtensor<float>(C);

  SubtensorList<const float> vAs = getVectorList(A);
  SubtensorList<const float> vBs = getVectorList(B);
  SubtensorList<float> vCs = getVectorList(Cs);
  CHECK(vAs.getSize() == vCs.getSize());
  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const float> vA = vAs.getSubtensor(j);
    Subtensor<const float> vB = vBs.getSubtensor(j % vBs.getSize());
    Subtensor<float> vC = vCs.getSubtensor(j);

    for (int i = 0; i < vA.dimension(0); ++i) {
      vC.elem(i) = vA.elem(i) + vB.elem(i);
    }
  }

  return C;
}

Tensor CPUOperators::softmaxFp32(Subtensor<const float> A) {
  Tensor C = createTensorLike(A);
  Subtensor<float> Cs = makeSubtensor<float>(C);

  auto softmax_op = [](Subtensor<const float> A, Subtensor<float> C) {
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

  SubtensorList<const float> vAs = getVectorList(A);
  SubtensorList<float> vCs = getVectorList(Cs);
  CHECK(vAs.getSize() == vCs.getSize());
  for (int i = 0; i < vAs.getSize(); ++i) {
    softmax_op(vAs.getSubtensor(i), vCs.getSubtensor(i));
  }

  return C;
}

Tensor CPUOperators::geluFp32(Subtensor<const float> A) {
  Tensor C = createTensorLike(A);
  Subtensor<float> Cs = makeSubtensor<float>(C);

  auto gelu_op = [](Subtensor<const float> A, Subtensor<float> C) {
    for (int i = 0; i < A.dimension(0); ++i) {
      float x = A.elem(i);

      double x3 = pow(x, 3.0);
      double c = 0.5 * x * (1 + tanh(sqrt(2.0 / kPi) * (x + 0.044715 * x3)));
      C.elem(i) = static_cast<float>(c);
    }
  };

  SubtensorList<const float> vAs = getVectorList(A);
  SubtensorList<float> vCs = getVectorList(Cs);
  CHECK(vAs.getSize() == vCs.getSize());
  for (int i = 0; i < vAs.getSize(); ++i) {
    gelu_op(vAs.getSubtensor(i), vCs.getSubtensor(i));
  }
  return C;
}

bool CPUOperators::allCloseFp32(
    Subtensor<const float> A, Subtensor<const float> B, float rtol, float atol) {
  CHECK(isShapeMatch(A, B));

  SubtensorList<const float> vAs = getVectorList(A);
  SubtensorList<const float> vBs = getVectorList(B);
  CHECK(vAs.getSize() == vBs.getSize());

  bool all_close = true;
  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const float> vA = vAs.getSubtensor(j);
    Subtensor<const float> vB = vBs.getSubtensor(j);

    for (int i = 0; i < vA.dimension(0); ++i) {
      float va = vA.elem(i);
      float vb = vB.elem(i);
      if (!(std::isfinite(va) && std::isfinite(vb))) {
        all_close = false;
      }
      if (fabs(va - vb) > atol + rtol * fabs(vb)) {
        all_close = false;
      }
    }
  }
  
  return all_close;
}

Tensor CPUOperators::mulFp32(Subtensor<const float> A, float k) {
  Tensor C = createTensorLike(A);
  Subtensor<float> Cs = makeSubtensor<float>(C);

  SubtensorList<const float> vAs = getVectorList(A);
  SubtensorList<float> vCs = getVectorList(Cs);
  CHECK(vAs.getSize() == vCs.getSize());
  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const float> vA = vAs.getSubtensor(j);
    Subtensor<float> vC = vCs.getSubtensor(j);

    for (int i = 0; i < vA.dimension(0); ++i) {
      vC.elem(i) = k * vA.elem(i);
    }
  }
  return C;
}

void CPUOperators::copyFp32(Subtensor<const float> src, Subtensor<float> tgt) {
  SubtensorList<const float> vAs = getVectorList(src);
  SubtensorList<float> vCs = getVectorList(tgt);
  CHECK(vAs.getSize() == vCs.getSize());
  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const float> vA = vAs.getSubtensor(j);
    Subtensor<float> vC = vCs.getSubtensor(j);

    for (int i = 0; i < vA.dimension(0); ++i) {
      vC.elem(i) = vA.elem(i);
    }
  }
}

Tensor CPUOperators::lookupFp32(Subtensor<const float> table, Subtensor<const LongType> indices) {
  CHECK(table.rank() == 2 && indices.rank() == 2);

  int batch_size = indices.dimension(0);
  int seq_len = indices.dimension(1);
  int d_model = table.dimension(1);
  Tensor output = createTensor(ly::makeConstSpan({batch_size, seq_len, d_model}), DType::kFloat);
  Subtensor<float> emb = makeSubtensor<float>(output);

  for (int batch = 0; batch < batch_size; ++batch) {
    Subtensor<const LongType> indices_b = indices.subtensor(batch);
    Subtensor<float> emb_b = emb.subtensor(batch);
    for (int l = 0; l < seq_len; ++l) {
      int64_t index = indices_b.elem(l);
      CHECK(index < table.dimension(0)) << "indices out of range";

      Subtensor<const float> emb_src = table.subtensor(static_cast<int>(index));
      Subtensor<float> emb_tgt = emb_b.subtensor(l);
      copyFp32(emb_src, emb_tgt);
    }
  }

  return output;
}

Tensor CPUOperators::layerNormFp32(
    Subtensor<const float> input,
    Subtensor<const float> weight,
    Subtensor<const float> bias,
    float eps) {
  CHECK(bias.rank() == 1 && weight.rank() == 1);
  CHECK(weight.dimension(0) == bias.dimension(0));
  CHECK(input.dimension(input.rank() - 1) == weight.dimension(0));

  Tensor C = createTensorLike(input);
  Subtensor<float> Cs = makeSubtensor<float>(C);
  SubtensorList<const float> vAs = getVectorList(input);
  SubtensorList<float> vCs = getVectorList(Cs);

  CHECK(vAs.getSize() == vCs.getSize());
  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const float> vA = vAs.getSubtensor(j);
    Subtensor<float> vC = vCs.getSubtensor(j);

    double sum = 0.0f;
    for (int i = 0; i < vA.dimension(0); ++i) {
      sum += vA.elem(i);
    }
    double mean = sum / vA.dimension(0);
    
    // var (unbiased)
    sum = 0.0;
    for (int i = 0; i < vA.dimension(0); ++i) {
      double d = vA.elem(i) - mean;
      sum += d * d;
    }
    double var = sum / vA.dimension(0);
    double sd = sqrt(var + eps);

    // compute layer-norm
    for (int i = 0; i < vA.dimension(0); ++i) {
      float elem = static_cast<float>((vA.elem(i) - mean) / sd); 
      vC.elem(i) = elem * weight.elem(i) + bias.elem(i);
    }
  }

  return C;
}

Tensor CPUOperators::causalMaskFp32(int seq_len) {
  Tensor mask = createTensor(ly::makeConstSpan({seq_len, seq_len}), DType::kFloat);
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

void CPUOperators::catFp32(Subtensor<const float> A, Subtensor<const float> B, int dim, Subtensor<float> C) {
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

  return F;
}

Tensor CPUOperators::createTensor(std::initializer_list<int> shape, DType dtype) {
  return createTensor(ly::makeConstSpan(shape), dtype);
}

Tensor CPUOperators::createTensorLike(const Tensor &input) {
  std::vector<Tensor::ShapeType> shape_vec;
  for (const TensorShape::Elem &elem : input._shape._data) {
    shape_vec.push_back(elem.shape);
  }

  return createTensor(ly::makeConstSpan(shape_vec), input.getDType());
}

Tensor CPUOperators::rand(std::initializer_list<int> shape, DType dtype) {
  Tensor tensor = createTensor(shape, dtype);
  switch (dtype) {
    case DType::kFloat:
      randFp32(&tensor);
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
      zerosFp32(makeSubtensor<float>(tensor));
      break;
    default:
      CHECK(false) << "unsupported dtype for Zeros";
  }

  return tensor;
}

Tensor CPUOperators::matmul(const Tensor &A, const Tensor &B) {
  switch (A.getDType()) {
    case DType::kFloat:
      return matmulFp32(A, B);
      break;
    default:
      CHECK(false) << "unsupported dtype for MatMul";
      return Tensor();
  }
}

void CPUOperators::print(const Tensor &tensor) {
  switch (tensor.getDType()) {
    case DType::kFloat:
      printFp32(makeConstSubtensor<float>(tensor));
      break;
    default:
      CHECK(false) << "unsupported dtype for Print";
  }
}

Tensor CPUOperators::add(const Tensor &input, const Tensor &other) {
  switch (input.getDType()) {
    case DType::kFloat:
      return addFp32(makeConstSubtensor<float>(input), makeConstSubtensor<float>(other));
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::softmax(const Tensor &input) {
  switch (input.getDType()) {
    case DType::kFloat:
      return softmaxFp32(makeConstSubtensor<float>(input));
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::gelu(const Tensor &input) {
  switch (input.getDType()) {
    case DType::kFloat:
      return geluFp32(makeConstSubtensor<float>(input));
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
      return allCloseFp32(makeConstSubtensor<float>(A), makeConstSubtensor<float>(B), 1e-6f, 1e-3f);
      break;
    default:
      NOT_IMPL();
  }

  return false;
}

Tensor CPUOperators::mul(const Tensor &A, float k) {
  switch (A.getDType()) {
    case DType::kFloat:
      return mulFp32(makeConstSubtensor<float>(A), k);
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
        copyFp32(makeConstSubtensor<float>(input), makeSubtensor<float>(C));
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
      return lookupFp32(makeConstSubtensor<float>(table), makeConstSubtensor<LongType>(indices));
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
      return layerNormFp32(
          makeConstSubtensor<float>(input),
          makeConstSubtensor<float>(weight),
          makeConstSubtensor<float>(bias),
          eps);
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::causalMask(int max_len) {
  return causalMaskFp32(max_len);
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

  Tensor C = createTensor(ly::makeConstSpan(shape), A.getDType());
  switch (A.getDType()) {
    case DType::kFloat:
      catFp32(
          makeConstSubtensor<float>(A),
          makeConstSubtensor<float>(B),
          dim,
          makeSubtensor<float>(C));
      break;
    default:
      NOT_IMPL();
  }

  return C;
}

}  // namespace flint
