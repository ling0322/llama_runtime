#ifndef LLAMA_CC_TENSOR_H_
#define LLAMA_CC_TENSOR_H_

#include <array>
#include <complex>
#include <type_traits>
#include <vector>

#include "common.h"
#include "status.h"

namespace llama {

class ReadableFile;
class Tensor;
class TensorView;

template <typename T>
class TensorBase1D {
 public:
  typedef T data_type;

  // destructor
  virtual ~TensorBase1D();

  // functions to access metedata
  int shape0() const { return shape0_; }
  bool empty() const { return !data_; }

  // functions to access elements
  T *data() const { return data_; }
  T *begin() const { return data_; }
  T *end() const { return data_ + shape0_; }
  T &operator[](int index) const { return data_[index]; }

  // copy the data from another tensor. They should be in the same shape
  void CopyFrom(const TensorBase1D<T> &tensor);

  // number of element in this tensor
  int NumEl() const;

  // apply x^2 to all elements
  void ApplySquare();

  // apply log10 to all elements
  void ApplyLog10();

  // return the max value in this tensor
  T Max() const;

  // multiply scalar value
  void Mul(T scalar);

  // add scalar value
  void Add(T scalar);

  // clamps all elements in tensor into range
  void ApplyMinClamp(T min);

  // apply log_softmax
  void ApplyLogSoftmax();

  // apply element-wise multiple with another tensor and store result to itself
  void ElementMul(const TensorBase1D<T> &tensor);

  // Print the tensor for debugging
  void DebugPrint(bool print_shape = true);

  // convert to dynamic tensor view
  const TensorView ToTensorView() const;

 protected:
  T *data_;
  int shape0_;

  TensorBase1D();
  TensorBase1D(T *data, int shape0);
};

// 1D tensor (Vector).
template <typename T>
class TensorView1D : public TensorBase1D<T> {
 public:
  // create Tensor1D from raw pointer and shape
  static TensorView1D<T> From(T *data, int shape0);
};

// 1D tensor, owns the data.
template <typename T>
class Tensor1D : public TensorBase1D<T> {
 public:
  Tensor1D();
  Tensor1D(int shape0);
  Tensor1D(Tensor1D<T> &&tensor);
  ~Tensor1D();

  Tensor1D<T> &operator=(Tensor1D<T> &&tensor);

  // resize the tensor, original data may loss after calling this function
  void Resize(int shape0);
  TensorView1D<T> View();

 private:
  DISALLOW_COPY_AND_ASSIGN(Tensor1D);
};

template <typename T>
class TensorView2D;

template <typename T>
class TensorBase2D {
 public:
  typedef T data_type;

  TensorBase2D();
  virtual ~TensorBase2D();

  int shape0() const { return shape0_; }
  int shape1() const { return shape1_; }
  bool empty() const { return !data_; }
  const T *data() const { return data_; }
  TensorView1D<T> operator[](int index0) const;

  // Get slice of this tensor along dimension 0
  TensorView2D<T> Slice(int start, int stop) const;

  // number of element in this tensor
  int NumEl() const;

  T Max() const;
  void Mul(T scalar);
  void Add(T scalar);
  void Add(T alpha, const TensorBase2D<T> &tensor);
  void CopyFrom(const TensorBase2D<T> &tensor);
  void ApplyAbs();
  void ApplyMinClamp(T min);

  // Print the tensor for debugging
  void DebugPrint(bool print_shape = true);

  // convert to dynamic tensor view
  const TensorView ToTensorView() const;

 protected:
  T *data_;
  int shape0_;
  int shape1_;

  TensorBase2D(T *data, int shape0, int shape1);
};

template<typename T>
class TensorView2D : public TensorBase2D<T> {
 public:
  // create Tensor1D from raw pointer and shape
  static TensorView2D<T> From(T *data, int shape0, int shape1);
};

// 2D tensor, owns the data.
template<typename T>
class Tensor2D : public TensorBase2D<T> {
 public:
  Tensor2D();
  Tensor2D(int shape0, int shape1);
  Tensor2D(Tensor2D<T> &&tensor) noexcept;
  Tensor2D(Tensor &&tensor);
  ~Tensor2D();

  Tensor2D<T> &operator=(Tensor2D<T> &&tensor);

  // resize the tensor, original data may loss after calling this function
  void Resize(int shape0, int shape1);
  TensorView2D<T> View();

 private:
  DISALLOW_COPY_AND_ASSIGN(Tensor2D);
};

typedef Tensor1D<float> Tensor1Df;
typedef Tensor1D<int64_t> Tensor1Dl;
typedef Tensor1D<std::complex<float>> Tensor1Dc;
typedef Tensor2D<float> Tensor2Df;
typedef TensorView1D<float> TensorView1Df;
typedef TensorView1D<int64_t> TensorView1Dl;
typedef TensorView1D<std::complex<float>> TensorView1Dc;
typedef TensorView2D<float> TensorView2Df;
typedef TensorView2D<std::complex<float>> TensorView2Dc;

// dynamic-dimension, dynamic-dtype tensor view. Reshape() supported.
class TensorBase {
 public:
  // max dimension supported in TensorView
  static constexpr int kMaxRank = 3;

  virtual ~TensorBase();

  int rank() const { return rank_; }
  DType dtype() const { return dtype_; }
  int shape(int d) const;
  // pointer to the begining of shape array
  const int32_t *shape_data() const { return shape_.data(); }
  int numel() const;

  // returns the data pointer as T *
  void *raw_data() { return data_; }
  const void *raw_data() const { return data_; }
  template <typename T>
  T *data() {
    ABSL_CHECK(TypeID<T>() == dtype_);
    return data_;
  }

  // check the dimension and dtype of this tensor, throw errors if they did not
  // match
  Status EnsureRankDType(int dim, DType dtype);

  // convert to fixed-dimension, fixed-dtype tensor
  TensorView2Df ToTensorView2Df();
  const TensorView2Df ToTensorView2Df() const;

 protected:
  std::array<int32_t, kMaxRank> shape_;
  int16_t rank_;
  DType dtype_;  // int16_t
  void *data_;

  TensorBase();
  TensorBase(std::initializer_list<int32_t> shape, DType dtype, void *data);
  TensorBase(const int *shape, int rank, DType dtype, void *data);
};

class TensorView : public TensorBase {
 public:
  static TensorView From(std::initializer_list<int32_t> shape,
                         DType dtype,
                         void *data);
  static TensorView From(const int *shape,
                         int rank,
                         DType dtype,
                         void *data);
};

// ----------------------------------------------------------------------------
// class Tensor
// ----------------------------------------------------------------------------

// dynamic-dimension, dynamic-dtype tensor view. Supports Reshape. Also owns
// the underlying data
class Tensor : public TensorBase {
 public:
  Tensor();
  Tensor(Tensor &&);
  Tensor(DType dtype, std::initializer_list<int32_t> shape);
  ~Tensor();

  template<typename T>
  Tensor2D<T> MoveAsTensor2D() &&;

  // read the tensor from stream
  Status Read(ReadableFile *fp);
  TensorView View();

 private:
  template <typename T>
  friend class Tensor2D;

  DISALLOW_COPY_AND_ASSIGN(Tensor);
};

template<typename T>
Tensor2D<T> Tensor::MoveAsTensor2D() && {
  ASSERT(rank_ == 2 && TypeID<T>() == dtype_);
  Tensor2D<T> tensor;

  tensor.data_ = reinterpret_cast<T *>(data_);
  data_ = nullptr;

  tensor.shape0_ = shape_[0];
  tensor.shape1_ = shape_[1];

  rank_ = 0;
  dtype_ = DType::kUnknown;

  return tensor;
}

// ----------------------------------------------------------------------------
// namespace tensor
// ----------------------------------------------------------------------------

namespace tensor {

template <typename T, class I>
Tensor2D<T> Stack1D(I begin, I end);

template <class I>
Tensor2D<float> Stack1Df(I begin, I end) {
  return Stack1D<float>(begin, end);
}

}  // namespace tensor
}  // namespace llama

#endif  // LLAMA_CC_TENSOR_H_
