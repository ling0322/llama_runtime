#include "tensor.h"

#include <stdio.h>
#include <cmath>
#include <deque>
#include <complex>
#include <limits>
#include <memory>
#include <vector>
#include "common.h"
#include "io.h"
#include "log.h"

namespace llama {

// --------------------------- TensorBase1D ------------------------------------

template<typename T>
TensorBase1D<T>::TensorBase1D(T *data, int shape0):
    data_(data), shape0_(shape0) {}

template<typename T>
TensorBase1D<T>::TensorBase1D(): data_(nullptr), shape0_(0) {}

template<typename T>
TensorBase1D<T>::~TensorBase1D() {}

template<typename T>
void TensorBase1D<T>::CopyFrom(const TensorBase1D<T> &tensor) {
  CHECK(shape0() == tensor.shape0());
  std::copy(tensor.begin(), tensor.end(), begin());
}

template<typename T>
int TensorBase1D<T>::NumEl() const {
  return shape0_;
}

template<typename T>
T TensorBase1D<T>::Max() const {
  CHECK(!empty());
  T max = *data_;
  for (int i = 0; i < NumEl(); ++i) {
    max = std::max(max, data_[i]);
  }

  return max;
}

template<typename T>
void TensorBase1D<T>::Mul(T scalar) {
  for (int i = 0; i < NumEl(); ++i) {
    data_[i] *= scalar;
  }
}

template<typename T>
void TensorBase1D<T>::Add(T scalar) {
  for (int i = 0; i < NumEl(); ++i) {
    data_[i] += scalar;
  }
}

template<typename T>
void TensorBase1D<T>::ApplySquare() {
  for (int i = 0; i < shape0(); ++i) {
    T val = data_[i];
    data_[i] = val * val;
  }
}

template<typename T>
void TensorBase1D<T>::ApplyMinClamp(T min) {
  for (int i = 0; i < shape0(); ++i) {
    T val = data_[i];
    data_[i] = val < min ? min : val;
  }
}

// x(i) = x(i) - log(sum(exp(x(i))))
template<typename T>
void TensorBase1D<T>::ApplyLogSoftmax() {
  T sum = 0;
  for (int i = 0; i < shape0(); ++i) {
    sum += std::exp(data_[i]);
  }
  T logsum = std::log(sum);
  Add(T(-1.0) * logsum);
}

template<typename T>
void TensorBase1D<T>::ElementMul(const TensorBase1D<T> &tensor) {
  CHECK(shape0() == tensor.shape0());
  for (int i = 0; i < shape0(); ++i) {
    data_[i] *= tensor.data_[i];
  }
}

template<typename T>
void TensorBase1D<T>::ApplyLog10() {
  for (int i = 0; i < shape0(); ++i) {
    data_[i] = std::log10(data_[i]);
  }
}

template<typename T>
void TensorBase1D<T>::DebugPrint(bool print_shape) { NOT_IMPL(); }
template<>
void TensorBase1D<float>::DebugPrint(bool print_shape) {
  if (print_shape) {
    printf("Tensor1D(");
  }
  printf("[");
  for (int i = 0; i < shape0(); ++i) {
    float elem = data_[i];
    if (std::abs(elem) > 100 || std::abs(elem) < 0.01) {
      printf("%.4e", data_[i]);
    } else {
      printf("%.4f", data_[i]);
    }

    if (shape0() > DEBUG_EDGE_ITEMS * 2 && i == DEBUG_EDGE_ITEMS - 1) {
      printf(" ... ");
      i += shape0() - DEBUG_EDGE_ITEMS * 2;
    } else if (i < shape0() - 1) {
      printf(", ");
    }
  }
  printf("]");
  if (print_shape) {
    printf(", shape=(%d,))\n", shape0());
  }
}

template<typename T>
const TensorView TensorBase1D<T>::ToTensorView() const {
  return TensorView::From({shape0_}, TypeID<T>(), data_);
}

// these following functions did not support std::complex
template<>
void TensorBase1D<std::complex<float>>::ApplySquare() { NOT_IMPL(); }
template<>
void TensorBase1D<std::complex<float>>::ApplyLog10() { NOT_IMPL(); }
template<>
const TensorView TensorBase1D<std::complex<float>>::ToTensorView() const {
  NOT_IMPL();
}
template<>
std::complex<float> TensorBase1D<std::complex<float>>::Max() const { 
  NOT_IMPL();
}
template<>
void TensorBase1D<int64_t>::ApplyMinClamp(int64_t min) { NOT_IMPL(); }
template<>
void TensorBase1D<std::complex<float>>::ApplyMinClamp(
    std::complex<float> min) { NOT_IMPL(); }

// --------------------------- TensorView1D ------------------------------------

template<typename T>
TensorView1D<T> TensorView1D<T>::From(T *data, int shape0) {
  TensorView1D<T> tensor;

  tensor.data_ = data;
  tensor.shape0_ = shape0;
  return tensor;
}

// ---------------------------- Tensor1D -------------------------------------

template <typename T>
Tensor1D<T>::Tensor1D(): TensorBase1D<T>(nullptr, 0) {};

template <typename T>
Tensor1D<T>::Tensor1D(int shape0):
    TensorBase1D<T>(shape0 ? new T[shape0] : nullptr, shape0) {}

template <typename T>
Tensor1D<T>::Tensor1D(Tensor1D<T> &&tensor):
    TensorBase1D<T>(tensor.data_, tensor.shape0_) {
  tensor.data_ = nullptr;
  tensor.shape0_ = 0;
}

template <typename T>
Tensor1D<T>::~Tensor1D() {
  delete[] TensorBase1D<T>::data_;
  TensorBase1D<T>::data_ = nullptr;

  TensorBase1D<T>::shape0_ = 0;
}

template <typename T>
Tensor1D<T> &Tensor1D<T>::operator=(Tensor1D<T> &&tensor) {
  delete[] TensorBase1D<T>::data_;

  TensorBase1D<T>::data_ = tensor.data_;
  TensorBase1D<T>::shape0_ = tensor.shape0_;
  tensor.data_ = nullptr;
  tensor.shape0_ = 0;

  return *this;
}

template <typename T>
void Tensor1D<T>::Resize(int shape0) {
  delete[] TensorBase1D<T>::data_;
  TensorBase1D<T>::data_ = shape0 ? new T[shape0] : nullptr;
  TensorBase1D<T>::shape0_ = shape0;
}

template <typename T>
TensorView1D<T> Tensor1D<T>::View() {
  return TensorView1D<T>::From(data_, shape0_);
}

// --------------------------- TensorBase2D -----------------------------------

template<typename T>
TensorBase2D<T>::TensorBase2D(T *data, int shape0, int shape1):
    data_(data), shape0_(shape0), shape1_(shape1) {}

template<typename T>
TensorBase2D<T>::TensorBase2D(): data_(nullptr), shape0_(0), shape1_(0) {}

template<typename T>
TensorBase2D<T>::~TensorBase2D() {}


template<typename T>
int TensorBase2D<T>::NumEl() const {
  return shape0_ * shape1_; 
}

template<typename T>
T TensorBase2D<T>::Max() const {
  CHECK(!empty());
  T max = *data_;
  for (int i = 0; i < NumEl(); ++i) {
    max = std::max(max, data_[i]);
  }

  return max;
}

template<typename T>
void TensorBase2D<T>::Mul(T scalar) {
  for (int i = 0; i < NumEl(); ++i) {
    data_[i] *= scalar;
  }
}

template<typename T>
void TensorBase2D<T>::Add(T scalar) {
  for (int i = 0; i < NumEl(); ++i) {
    data_[i] += scalar;
  }
}

template<typename T>
void TensorBase2D<T>::Add(T alpha, const TensorBase2D<T> &tensor) {
  CHECK(tensor.shape0() == shape0() && tensor.shape1() == shape1());
  for (int i = 0; i < NumEl(); ++i) {
    data_[i] += alpha * tensor.data_[i];
  }
}

template<typename T>
void TensorBase2D<T>::ApplyAbs() {
  for (int i = 0; i < NumEl(); ++i) {
    data_[i] = std::abs(data_[i]);
  }
}

template<typename T>
void TensorBase2D<T>::ApplyMinClamp(T min) {
  for (int i = 0; i < NumEl(); ++i) {
    T &elem = data_[i];
    if (elem < min) {
      elem = min;
    }
  }
}

template<typename T>
TensorView1D<T> TensorBase2D<T>::operator[](int index0) const {
  return TensorView1D<T>::From(data_ + index0 * shape1_, shape1_);
}

template<typename T>
TensorView2D<T> TensorBase2D<T>::Slice(int start, int stop) const {
  int new_shape0 = stop - start;
  CHECK(new_shape0 > 0);

  return TensorView2D<T>::From(data_ + start * shape1_, new_shape0, shape1_);
}

template<typename T>
void TensorBase2D<T>::CopyFrom(const TensorBase2D<T> &tensor) {
  CHECK(shape0() == tensor.shape0());
  CHECK(shape1() == tensor.shape1());

  for (int i = 0; i < shape0(); ++i) {
    (*this)[i].CopyFrom(tensor[i]);
  }
}

template<typename T>
void TensorBase2D<T>::DebugPrint(bool print_shape) { NOT_IMPL(); }
template<>
void TensorBase2D<float>::DebugPrint(bool print_shape) {
  if (print_shape) {
    printf("Tensor2D(");
  }
  printf("[");
  for (int i = 0; i < shape0(); ++i) {
    float elem = data_[i];
    if (i > 0) {
      printf("          ");
    }
    (*this)[i].DebugPrint(false);
    if (i < shape0() - 1) {
        printf(",\n");
    }

    if (shape0() > DEBUG_EDGE_ITEMS * 2 && i == DEBUG_EDGE_ITEMS - 1) {
      printf("          ...\n");
      i += shape0() - DEBUG_EDGE_ITEMS * 2;
    }
  }
  printf("]");
  if (print_shape) {
    printf(", shape=(%d, %d))\n", shape0(), shape1());
  }
}

// these following functions did not support std::complex
template<>
std::complex<float> TensorBase2D<std::complex<float>>::Max() const { NOT_IMPL(); }
template<>
void TensorBase2D<std::complex<float>>::ApplyAbs() { NOT_IMPL(); }
template<>
void TensorBase2D<std::complex<float>>::ApplyMinClamp(std::complex<float> min) {
  NOT_IMPL(); 
}


// --------------------------- TensorView2D -----------------------------------

template<typename T>
TensorView2D<T> TensorView2D<T>::From(T *data, int shape0, int shape1) {
  TensorView2D<T> tensor;

  tensor.data_ = data;
  tensor.shape0_ = shape0;
  tensor.shape1_ = shape1;
  return tensor;
}

// ---------------------------- Tensor2D -------------------------------------

template <typename T>
Tensor2D<T>::Tensor2D(): TensorBase2D<T>(nullptr, 0, 0) {};

template <typename T>
Tensor2D<T>::Tensor2D(int shape0, int shape1):
    TensorBase2D<T>(
        shape0 * shape1 ? new T[shape0 * shape1] : nullptr,
        shape0,
        shape1) {}

template <typename T>
Tensor2D<T>::Tensor2D(Tensor2D<T> &&tensor) noexcept:
    TensorBase2D<T>(tensor.data_, tensor.shape0_, tensor.shape1_) {
  tensor.data_ = nullptr;
  tensor.shape0_ = 0;
  tensor.shape1_ = 0;
}

template <typename T>
Tensor2D<T>::Tensor2D(Tensor &&tensor) {
  CHECK(tensor.rank_ == 2);
  CHECK(tensor.dtype_ == TypeID<T>());

  TensorBase2D<T>::data_ = reinterpret_cast<T *>(tensor.data_);
  TensorBase2D<T>::shape0_ = tensor.shape_[0];
  TensorBase2D<T>::shape1_ = tensor.shape_[1];

  tensor.data_ = nullptr;
  tensor.rank_ = 0;
  tensor.dtype_ = DType::kUnknown;
}

template <typename T>
Tensor2D<T>::~Tensor2D() {
  delete[] TensorBase2D<T>::data_;
  TensorBase2D<T>::data_ = nullptr;

  TensorBase2D<T>::shape0_ = 0;
  TensorBase2D<T>::shape1_ = 0;
}

template <typename T>
Tensor2D<T> &Tensor2D<T>::operator=(Tensor2D<T> &&tensor) {
  delete[] TensorBase2D<T>::data_;

  TensorBase2D<T>::data_ = tensor.data_;
  TensorBase2D<T>::shape0_ = tensor.shape0_;
  TensorBase2D<T>::shape1_ = tensor.shape1_;
  tensor.data_ = nullptr;
  tensor.shape0_ = 0;
  tensor.shape1_ = 0;

  return *this;
}

template <typename T>
void Tensor2D<T>::Resize(int shape0, int shape1) {
  delete[] TensorBase2D<T>::data_;
  TensorBase2D<T>::data_ = shape0 * shape1 ? new T[shape0 * shape1] : nullptr;
  TensorBase2D<T>::shape0_ = shape0;
  TensorBase2D<T>::shape1_ = shape1;
}

template <typename T>
TensorView2D<T> Tensor2D<T>::View() {
  return TensorView2D<T>::From(data_, shape0_, shape1_);
}

// ------------------------------- TensorBase --------------------------------

TensorBase::TensorBase()
    : rank_(0), dtype_(DType::kUnknown), data_ (nullptr) {}

TensorBase::TensorBase(
    std::initializer_list<int32_t> shape,
    DType dtype,
    void *data)
      : rank_(static_cast<int16_t>(shape.size())),
        dtype_(dtype),
        data_(data) {
  CHECK(shape.size() < kMaxRank);
  std::copy(shape.begin(), shape.end(), shape_.begin());
}

TensorBase::TensorBase(const int *shape, int rank, DType dtype, void *data)
    : rank_(rank),
      dtype_(dtype),
      data_(data){
  CHECK(rank < kMaxRank);
  std::copy(shape, shape + rank, shape_.begin());
}

TensorBase::~TensorBase() {}

int TensorBase::shape(int d) const {
  CHECK(d < rank_);
  return shape_[d];
}

int TensorBase::numel() const {
  int n = 1;
  for (int i = 0; i < rank_; ++i) {
    n *= shape_[i];
  }

  return n;
}

Status TensorBase::EnsureRankDType(int rank, DType dtype) {
  if (rank != rank_) {
    RETURN_ABORTED() << "rank mismatch";
  }
  if (dtype != dtype_) {
    RETURN_ABORTED() << "dtype mismatch";
  }

  return OkStatus();
}

const TensorView2Df TensorBase::ToTensorView2Df() const {
  CHECK(rank_ == 2);
  CHECK(dtype_ == TypeID<float>());

  return TensorView2Df::From(reinterpret_cast<float *>(data_),
                             shape_[0],
                             shape_[1]);
}

TensorView2Df TensorBase::ToTensorView2Df() {
  CHECK(rank_ == 2);
  CHECK(dtype_ == TypeID<float>());

  return TensorView2Df::From(reinterpret_cast<float *>(data_),
                             shape_[0],
                             shape_[1]);
}

// ------------------------------- TensorView --------------------------------

TensorView TensorView::From(
    std::initializer_list<int32_t> shape,
    DType dtype,
    void* data) {
  TensorView view;
  view.rank_ = static_cast<int16_t>(shape.size());
  view.dtype_ = dtype;
  view.data_ = data;

  CHECK(shape.size() < kMaxRank);
  std::copy(shape.begin(), shape.end(), view.shape_.begin());

  return view;
}

TensorView TensorView::From(
    const int *shape,
    int rank,
    DType dtype,
    void *data) {
  TensorView view;
  view.rank_ = rank;
  view.dtype_ = dtype;
  view.data_ = data;

  CHECK(rank < kMaxRank);
  std::copy(shape, shape + rank, view.shape_.begin());

  return view;
}

// -------------------------------- Tensor -----------------------------------

Tensor::Tensor(): TensorBase({}, DType::kUnknown, nullptr) {}

Tensor::Tensor(DType dtype, std::initializer_list<int32_t> shape):
    TensorBase(shape, dtype, nullptr) {
  int size = 1;
  for (int32_t n : shape) {
    size *= n;
  }

  int n_bytes = size * SizeOfDType(dtype);
  data_ = new ByteType[n_bytes];
}

Tensor::Tensor(Tensor &&tensor) : TensorBase({}, tensor.dtype_, tensor.data_) {
  std::copy(tensor.shape_.begin(),
            tensor.shape_.begin() + tensor.rank_,
            shape_.begin());
  
  tensor.data_ = nullptr;
  tensor.rank_ = 0;
  tensor.dtype_ = DType::kUnknown;
}

Tensor::~Tensor() {
  delete[] data_;
  data_ = nullptr;
}

Status Tensor::Read(ReadableFile *fp) {
  // clear original data
  delete[] data_;
  data_ = nullptr;
  dtype_ = DType::kUnknown;
  rank_ = 0;

  std::string s;
  RETURN_IF_ERROR(fp->ReadString(4, &s));
  if (s != "TNSR") {
    RETURN_ABORTED() << "bad tensor format";
  }

  // dimension
  int16_t dimension;
  RETURN_IF_ERROR(fp->ReadValue(&dimension));
  if (dimension > kMaxRank) {
    RETURN_ABORTED() << "too many dimensions";
  }
  rank_ = dimension;

  // data type
  int16_t dtype;
  RETURN_IF_ERROR(fp->ReadValue(&dtype));
  if (!IsValidDType(static_cast<DType>(dtype))) {
    RETURN_ABORTED() << "invalid dtype";
  }
  dtype_ = static_cast<DType>(dtype);

  // shape
  int numel = 1;
  int32_t size;
  for (int d = 0; d < rank_; ++d) {
    RETURN_IF_ERROR(fp->ReadValue(&size));

    shape_[d] = size;
    numel *= size;
  }
  if (numel > 4194304) {
    RETURN_ABORTED() << "tensor too big";
  }

  int byte_size = numel * SizeOfDType(dtype_);
  data_ = new ByteType[byte_size];
  util::Span<ByteType> bs_data(reinterpret_cast<ByteType *>(data_), byte_size);
  RETURN_IF_ERROR(fp->ReadSpan(bs_data));

  return OkStatus();
}

TensorView Tensor::View() {
  return TensorView::From(shape_.data(), rank_, dtype_, data_);
}

// ------------------------- template instantiation --------------------------

template class Tensor1D<float>;
template class Tensor1D<std::complex<float>>;
template class Tensor2D<float>;
template class TensorBase1D<float>;
template class TensorBase1D<std::complex<float>>;
template class TensorBase1D<float>;
template class TensorView1D<float>;
template class TensorView1D<std::complex<float>>;
template class TensorBase2D<float>;
template class TensorView2D<float>;

// --------------------------- tensor:: --------------------------------------

namespace tensor {

template<typename T, typename I>
Tensor2D<T> Stack1D(I begin, I end) {
  int shape0 = static_cast<int>(end - begin);
  if (shape0 <= 0) {
    return Tensor2D<T>();
  }

  Tensor2D<T> tensor2d(shape0, begin->shape0());
  int i = 0;
  for (I it = begin; it < end; ++it) {
    CHECK(it->shape0() == tensor2d.shape1());
    tensor2d[static_cast<int>(it - begin)].CopyFrom(*it);
  }

  return tensor2d;
}

typedef std::deque<Tensor1D<float>>::iterator IterDeqT1D;
typedef std::vector<Tensor1D<float>>::iterator IterVecT1D;
template Tensor2D<float> Stack1D(IterDeqT1D, IterDeqT1D);
template Tensor2D<float> Stack1D(IterVecT1D, IterVecT1D);

}  // namespace tensor

}  // namespace llama
