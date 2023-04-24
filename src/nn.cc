#include "nn.h"

#include <stdlib.h>
#include <limits>
#include "gemm.h"

namespace llama {
namespace nn {

Tensor::Tensor() : shape_(nullptr),
                   data_(nullptr),
                   dtype_(DType::kUnknown),
                   rank_(kEmptyRank),
                   flag_(0) {}

Tensor::~Tensor() {
  Dispose();
}

Tensor::Tensor(Tensor &&tensor) {
  shape_ = tensor.shape_;
  data_ = tensor.data_;
  dtype_ = tensor.dtype_;
  rank_ = tensor.rank_;
  flag_ = tensor.flag_;

  tensor.shape_ = nullptr;
  tensor.data_ = nullptr;
  tensor.dtype_ = DType::kUnknown;
  tensor.rank_ = kEmptyRank;
  tensor.flag_ = 0;
}

Tensor &Tensor::operator=(Tensor &&tensor) {
  Dispose();

  shape_ = tensor.shape_;
  data_ = tensor.data_;
  dtype_ = tensor.dtype_;
  rank_ = tensor.rank_;
  flag_ = tensor.flag_;

  tensor.shape_ = nullptr;
  tensor.data_ = nullptr;
  tensor.dtype_ = DType::kUnknown;
  tensor.rank_ = kEmptyRank;
  tensor.flag_ = 0;

  return *this;
}

void Tensor::Dispose() {
  if (flag_ & kFlagOwnData) {
    delete[] data_;
  }

  if (flag_ & kFlagOwnShape) {
    delete[] shape_;
  }

  shape_ = nullptr;
  data_ = nullptr;
  dtype_ = DType::kUnknown;
  rank_ = kEmptyRank;
  flag_ = 0;
}

int Tensor::shape(int d) const {
  CHECK(!empty());
  int rank = rank_;
  if (d < 0) {
    d = rank + d;
  }

  CHECK(d >= 0 && d < rank);
  return shape_[d];
}

int Tensor::stride(int d) const {
  CHECK(!empty());
  int rank = rank_;
  if (d < 0) {
    d = rank + d;
  }

  CHECK(d >= 0 && d < rank);
  return shape_[rank + d];
}

int Tensor::numel() const {
  if (empty()) {
    return 0;
  } else {
    return rank_ ? shape_[0] * shape_[rank_]
                 : 1;
  }
}

bool Tensor::empty() const {
  return data_ == nullptr;
}

// ---------------------------------------------------------------------------+
// class Function                                                             |
// ---------------------------------------------------------------------------+

constexpr int kPrintEdgeItems = 3;

namespace {

void Rand_Float32(Tensor *tensor) {
  float *data = tensor->data<float>();
  int numel = tensor->numel();

  float randmax = RAND_MAX;
  for (int i = 0; i < numel; ++i) {
    data[i] = rand() / randmax;
  }
}

void Zeros_Float32(Tensor *tensor) {
  float *data = tensor->data<float>();
  int numel = tensor->numel();

  float randmax = RAND_MAX;
  for (int i = 0; i < numel; ++i) {
    data[i] = 0.0f;
  }
}

void MatMul_Float32(const Tensor &A, const Tensor &B, Tensor *C) {
  GEMM gemm;
  gemm.MatMul(
      A.shape(0), B.shape(1), A.shape(1),
      A.data<float>(), B.data<float>(), C->data<float>());
}

void Print1D_Float32(int d, const float *data) {
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

void Print2D_Float32(int m, int n, int lda, const float *data) {
  printf("[");
  for (int i = 0; i < m; ++i) {
    if (i > 0) {
      printf(" ");
    }
    Print1D_Float32(n, data + i * lda);
    
    if (i < m - 1) {
        printf(",\n");
    }

    if (m > kPrintEdgeItems * 2 && i == kPrintEdgeItems - 1) {
      printf(" ...\n");
      i += m - kPrintEdgeItems * 2;
    }
  }
  printf("]");
}

void Print_Float32(const Tensor &tensor) {
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

}  // anonymous namespace

Tensor Function::Tensor_(std::initializer_list<int> shape, DType dtype) {
  Tensor tensor;

  // rank
  int rank = shape.size();
  CHECK(rank <= Tensor::kMaxRank);
  tensor.rank_ = static_cast<Tensor::RankType>(rank);

  // shape
  if (rank) {
    tensor.shape_ = new Tensor::ShapeType[rank * 2];
    std::copy(shape.begin(), shape.end(), tensor.shape_);
  } else {
    tensor.shape_ = nullptr;
  }

  // stride
  int64_t stride = 1;
  for (int d = rank - 1; d >= 0; --d) {
    CHECK(stride < std::numeric_limits<Tensor::ShapeType>::max());
    tensor.shape_[rank + d] = static_cast<Tensor::ShapeType>(stride);
    stride *= tensor.shape_[d];
  }

  // dtype
  tensor.dtype_ = dtype;

  // data
  int numel = rank ? tensor.shape_[0] * tensor.shape_[rank] 
                   : 1;
  tensor.data_ = new ByteType[numel * SizeOfDType(dtype)];

  // flag
  tensor.flag_ |= Tensor::kFlagOwnData;
  tensor.flag_ |= Tensor::kFlagOwnShape;

  return tensor;
}

Tensor Function::Rand(std::initializer_list<int> shape, DType dtype) {
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

Tensor Function::Zeros(std::initializer_list<int> shape, DType dtype) {
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

Tensor Function::MatMul(const Tensor &A, const Tensor &B) {
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

void Function::Print(const Tensor &tensor) {
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
