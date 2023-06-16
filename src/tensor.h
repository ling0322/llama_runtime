#pragma once

#include <stdint.h>
#include "common.h"
#include "reader.h"
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
DType getTypeID();

// get the size of specific dtype
int getDTypeSize(DType dtype);

// return true of DType is valid
bool isValidDType(DType dtype);

// contains dimension and stride information for an axis in tensor
class TensorData {
 public:
  TensorData(int64_t numel, DType dtype);
  ~TensorData();

  ByteType *getData() const { return _data; }
  DType getDType() const { return _dtype; }
  int64_t getSizeInBytes() const { 
    return _numel * getDTypeSize(_dtype);
  }

 private:
  ByteType *_data;
  int64_t _numel;
  DType _dtype;
};

// Stores shape and stride of a Tensor.
class TensorShape {
 public:
  friend class CPUOperators;

  typedef int32_t ShapeType;
  struct Elem {
    ShapeType shape;
    ShapeType stride;
  };

  // an empty Tensor.
  TensorShape() = default;

  // from shape.
  TensorShape(util::Span<const ShapeType> shape);

  // Returns a Size that is a transposed version of current size. The given
  // dimensions dim0 and dim1 are swapped.
  TensorShape transpose(int dim0, int dim1) const;

  // add or remove one shape=1 dimension at specified dimension.
  TensorShape unsqueeze(int dim) const;
  TensorShape squeeze(int dim) const;

  // Returns a sub-Size starting at specified dimension.
  TensorShape subsize(int d) const;

  TensorShape(const TensorShape &size);
  TensorShape(TensorShape &&size) noexcept;
  TensorShape &operator=(const TensorShape &size);
  TensorShape &operator=(TensorShape &&size) noexcept;

  bool empty() const;
  int getDim() const;
  ShapeType getShape(int index) const;
  ShapeType getStride(int index) const;
  int64_t getNumEl() const;

  // set the value of shape(dim). Negative dim is allowed. By design, new `shape` should not
  // greater than shape(dim).
  void setShape(int dim, ShapeType shape);

  // convert negative dimension or index (in specific `dim`) to positive.
  int getRealDim(int dim) const;
  int getRealIndex(int dim, int index) const;

 private:
  util::FixedArray<Elem> _data;
};

class Tensor {
 public:
  friend class CPUOperators;

  // integer type for shape and stride
  typedef TensorShape::ShapeType ShapeType;

  // rank for empty tansor.
  static constexpr int kEmptyRank = -1;

  // Make a tensor in CPU.
  template<typename T>
  static Tensor create(std::initializer_list<int> shape, util::Span<const T> data);

  // create a tensor in CPU storage. Size of `data` should be the same as `shape.numel()`.
  // Example:
  //   Tensor x = Tensor::FromData({2, 2}, {1.0f, 0.8f, 0.6f, 0.2f});
  template<typename T>
  static Tensor create(util::Span<const int> shape, util::Span<const T> data);

  // constructor and destructor.
  Tensor();
  ~Tensor();

  // Read the tensor from fp.
  void read(ReadableFile *fp);

  // copy and move constructors.
  Tensor(const Tensor &tensor);
  Tensor &operator=(const Tensor &tensor);
  Tensor(Tensor &&tensor) noexcept;
  Tensor &operator=(Tensor &&tensor);

  // get numebr of dimentsions.
  int getDim() const { return _shape.getDim(); }

  // get the size in dimention `d`. `d` supports positive number (index) and negative number (index
  // from back). Crash if `d` is out of boundary
  ShapeType getShape(int d) const { return _shape.getShape(d); }
  std::vector<ShapeType> getShape() const;

  // get stride for dimension `d`. 
  ShapeType getStride(int d) const { return _shape.getStride(d); }

  // get number of elements in this tensor.
  int64_t getNumEl() const { return _shape.getNumEl(); }

  // return true if this tensor is empty.
  bool empty() const { return _shape.empty(); }

  // get data type.
  DType getDType() const;

  // return a new tensor with the same data as the self tensor but of a different shape.
  Tensor view(util::Span<const int> shape) const;

  // Get slice of this tensor. `dim` is the dimension to slice. [begin, end) is the range. For
  // [begin, end) only version, dimension 0 is used. Negative `begin` and `end` is accepted. Crash
  // if dim or range out of boundary.
  Tensor slice(int dim, int begin, int end) const;
  Tensor slice(int begin, int end) const;

  // Get subtensor at specified index of first dimension. Negative `index` is accepted. Crash if
  // `index` out of boundary.
  Tensor subtensor(int index) const;

  // add or remove an additional shape=1 dimension at specified position.
  Tensor unsqueeze(int dim) const;
  Tensor squeeze(int dim) const;

  Tensor transpose(int dim0, int dim1) const;

  // return true if the tensor is contigous.
  bool isContiguous() const;

  // pointer of data in this tensor
  template<typename T>
  T *getData() { 
    return reinterpret_cast<T *>(getRawData(getTypeID<T>())); 
  }
  template<typename T>
  const T *getData() const {
    return reinterpret_cast<T *>(getRawData(getTypeID<T>()));
  }

  // return specific element at index. Size of `indices` should be the same as tensor dimension.
  // And the data should in CPU.
  template<typename T>
  T getElem(util::Span<const int> indices);

  // Check the shape of a tensor. If shape of `tensor` does not match `shape`, return AbortedError
  // with message "invalid shape".
  void throwIfInvalidShape(std::initializer_list<int> shape);

 protected:
  std::shared_ptr<TensorData> _data;
  TensorShape _shape;
  ByteType *_dataPtr;

  // check dtype and return the point of underlying data
  ByteType *getRawData(DType dtype) const;
};

inline DType Tensor::getDType() const { 
  return _data ? _data->getDType() : DType::kUnknown;
}

template<typename T>
inline T Tensor::getElem(util::Span<const int> indices) {
  CHECK(indices.size() == getDim());

  const T *data = this->getData<T>();
  int64_t offset = 0;
  for (int d = 0; d < getDim(); ++d) {
    offset += indices[d] * getStride(d);
  }

  return data[offset];
}

typedef const Tensor &TensorCRef;

}  // namespace nn
}  // namespace llama
