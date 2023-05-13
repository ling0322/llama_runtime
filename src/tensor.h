#ifndef LLM_RUNTIME_TENSOR_H_
#define LLM_RUNTIME_TENSOR_H_

#include <stdint.h>
#include "common.h"
#include "reader.h"
#include "status.h"
#include "util.h"

namespace llama {
namespace nn {

typedef int64_t LongType;

enum class DType : int16_t { 
  kUnknown = 0,
  kFloat = 1,
  kLong = 2
};

// get type-id
template <typename T>
DType TypeID();

// get the size of specific dtype
int SizeOfDType(DType dtype);

// return true of DType is valid
bool IsValidDType(DType dtype);

// contains dimension and stride information for an axis in tensor
class TensorData {
 public:
  TensorData(int64_t numel, DType dtype);
  ~TensorData();

  ByteType *data() const { return data_; }
  DType dtype() const { return dtype_; }
  int64_t size_in_bytes() const { 
    return numel_ * SizeOfDType(dtype_);
  }

 private:
  ByteType *data_;
  int64_t numel_;
  DType dtype_;
};

// Stores shape and stride of a Tensor.
class Size {
 public:
  friend class CpuOperators;

  typedef int32_t ShapeType;
  struct Elem {
    ShapeType shape;
    ShapeType stride;
  };

  // an empty Tensor.
  Size() = default;

  // from shape.
  Size(util::Span<const ShapeType> shape);

  // Returns a Size that is a transposed version of current size. The given
  // dimensions dim0 and dim1 are swapped.
  Size Transpose(int dim0, int dim1) const;

  // Returns a sub-Size starting at specified dimension.
  Size Subsize(int d) const;

  Size(const Size &size);
  Size(Size &&size) noexcept;
  Size &operator=(const Size &size);
  Size &operator=(Size &&size) noexcept;

  bool empty() const;
  int dim() const;
  ShapeType shape(int index) const;
  ShapeType stride(int index) const;
  int64_t numel() const;

  // set the value of shape(dim). Negative dim is allowed. By design, new
  // `shape` should not greater than shape(dim).
  void set_shape(int dim, ShapeType shape);

  // convert negative dimension or index (in specific `dim`) to positive.
  int real_dim(int dim) const;
  int real_index(int dim, int index) const;

 private:
  util::FixedArray<Elem> data_;
};

class Tensor {
 public:
  friend class CpuOperators;

  // integer type for shape and stride
  typedef Size::ShapeType ShapeType;

  // rank for empty tansor.
  static constexpr int kEmptyRank = -1;

  // constructor and destructor.
  Tensor();
  ~Tensor();

  // Read the tensor from fp.
  Status Read(ReadableFile *fp);

  // copy and move constructors.
  Tensor(const Tensor &tensor);
  Tensor &operator=(const Tensor &tensor);
  Tensor(Tensor &&tensor) noexcept;
  Tensor &operator=(Tensor &&tensor);

  // get numebr of dimentsions.
  int dim() const { return size_.dim(); }

  // get the size in dimention `d`. `d` supports positive number
  // (index) and negative number (index from back). Crash if `d` is out of
  // boundary
  ShapeType shape(int d) const {
    return size_.shape(d);
  }

  // get stride for dimension `d`. 
  ShapeType stride(int d) const {
    return size_.stride(d);
  }

  // get number of elements in this tensor.
  int64_t numel() const {
    return size_.numel();
  }

  // return true if this tensor is empty.
  bool empty() const {
    return size_.empty();
  }

  // get data type.
  DType dtype() const;

  // return a new tensor with the same data as the self tensor but of a
  // different shape.
  Tensor View(std::initializer_list<int> shape) const;

  // Get slice of this tensor. `dim` is the dimension to slice. [begin, end) is
  // the range. For [begin, end) only version, dimension 0 is used. Negative
  // `begin` and `end` is accepted. Crash if dim or range out of boundary.
  Tensor Slice(int dim, int begin, int end) const;
  Tensor Slice(int begin, int end) const;

  // Get subtensor at specified index of first dimension. Negative `index` is
  // accepted. Crash if `index` out of boundary.
  Tensor Subtensor(int index) const;

  Tensor Transpose(int dim0, int dim1) const;

  // return true if the tensor is contigous.
  bool is_contiguous() const;

  // pointer of data in this tensor
  template <typename T>
  T *data() { 
    return reinterpret_cast<T *>(raw_data(TypeID<T>())); 
  }
  template <typename T>
  const T *data() const {
    return reinterpret_cast<T *>(raw_data(TypeID<T>()));
  }

 protected:
  std::shared_ptr<TensorData> data_;
  Size size_;
  ByteType *data_ptr_;

  // check dtype and return the point of underlying data
  ByteType *raw_data(DType dtype) const;
};

inline DType Tensor::dtype() const { 
  return data_ ? data_->dtype() : DType::kUnknown;
}

typedef const Tensor & CTensorRef;

}  // namespace nn
}  // namespace llama

#endif  // LLM_RUNTIME_TENSOR_H_
