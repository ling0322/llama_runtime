#ifndef LLM_RUNTIME_TENSOR_H_
#define LLM_RUNTIME_TENSOR_H_

#include <stdint.h>
#include "common.h"
#include "reader.h"
#include "status.h"
#include "util.h"

namespace llama {
namespace nn {

// contains dimension and stride information for an axis in tensor
class TensorData {
 public:
  TensorData(int numel, DType dtype);
  ~TensorData();

  ByteType *data() const { return data_; }
  DType dtype() const { return dtype_; }
  int64_t size_in_bytes() const { 
    return numel_ * SizeOfDType(dtype_);
  }

 private:
  ByteType *data_;
  int numel_;
  DType dtype_;
};

class Tensor {
 public:
  friend class CpuOperators;

  // integer type for shape and stride
  typedef int32_t ShapeType;

  // rank for empty tansor.
  static constexpr int kEmptyRank = -1;

  // constructor and destructor.
  Tensor();
  ~Tensor();

  // Read the tensor from fp.
  Status Read(ReadableFile *fp);

  // copy and move constructors.
  Tensor(Tensor &tensor);
  Tensor &operator=(Tensor &tensor);
  Tensor(Tensor &&tensor) noexcept;
  Tensor &operator=(Tensor &&tensor);

  // get numebr of dimentsions.
  int rank() const;

  // get the size in dimention `d`. `d` supports positive number
  // (index) and negative number (index from back). Crash if `d` is out of
  // boundary
  ShapeType shape(int d) const;

  // get stride for dimension `d`. 
  ShapeType stride(int d) const;

  // get number of elements in this tensor.
  int numel() const;

  // return true if this tensor is empty.
  bool empty() const;

  // get data type.
  DType dtype() const;

  Tensor View(std::initializer_list<int> shape) const;

  Tensor Transpose(int dim0, int dim1) const;

  // pointer of data in this tensor
  template <typename T>
  T *data() { return reinterpret_cast<T *>(raw_data(TypeID<T>())); }
  template <typename T>
  const T *data() const { return reinterpret_cast<T *>(raw_data(TypeID<T>())); }

 protected:
  struct Shape {
    ShapeType dimension;
    ShapeType stride;
  };

  std::shared_ptr<TensorData> data_;
  util::FixedArray<Shape> shape_;
  ByteType *data_ptr_;

  // check dtype and return the point of underlying data
  ByteType *raw_data(DType dtype) const;

  // convert negative dimension index to positive
  int real_dim(int dim) const;

  // fill shape and stride values in shape_ according to given `shape`
  void FillShapeStride(util::Span<const int> shape);
};

inline int Tensor::rank() const {
  return data_ptr_ ? shape_.size() : kEmptyRank;
}
inline bool Tensor::empty() const {
  return data_ptr_ == nullptr;
}
inline DType Tensor::dtype() const { 
  return data_ ? data_->dtype() : DType::kUnknown;
}

}  // namespace nn
}  // namespace llama

#endif  // LLM_RUNTIME_TENSOR_H_
