#include "nn.h"

#include <stdlib.h>
#include <limits>
#include "tensor.h"
#include "strings.h"

namespace llama {
namespace nn {

// ---------------------------------------------------------------------------+
// class DType                                                                |
// ---------------------------------------------------------------------------+

template <>
DType getTypeID<float>() {
  return DType::kFloat;
}
template <>
DType getTypeID<int64_t>() {
  return DType::kLong;
}

template DType getTypeID<float>();
template DType getTypeID<int64_t>();

int getDTypeSize(DType dtype) {
  switch (dtype) {
    case DType::kFloat:
      return 4;
    case DType::kLong:
      return 8;
    default:
      NOT_IMPL();
  }
}

bool IsValidDType(DType dtype) {
  switch (dtype) {
    case DType::kFloat:
    case DType::kLong:
      return true;
    default:
      return false;
  }
}

// -- class TensorShape ----------

TensorShape::TensorShape(const TensorShape &size) : _data(size._data.copy()) {}
TensorShape::TensorShape(TensorShape &&size) noexcept : _data(std::move(size._data)) {}
TensorShape &TensorShape::operator=(const TensorShape &size) {
  _data = size._data.copy();
  return *this;
}
TensorShape &TensorShape::operator=(TensorShape &&size) noexcept {
  _data = std::move(size._data);
  return *this;
}

TensorShape::TensorShape(util::Span<const ShapeType> shape) {
  _data = util::FixedArray<Elem>(shape.size());
  util::FixedArray<Elem>::iterator it = _data.begin();
  for (int n : shape) {
    it->shape = n;
    ++it;
  }

  int64_t stride = 1;
  for (int d = shape.size() - 1; d >= 0; --d) {
    CHECK(stride < std::numeric_limits<ShapeType>::max());
    _data[d].stride = static_cast<ShapeType>(stride);
    stride *= _data[d].shape;
  }
}

TensorShape TensorShape::subsize(int d) const {
  CHECK(d < getDim());

  TensorShape subsize;
  subsize._data = util::FixedArray<Elem>(getDim() - d);
  std::copy(_data.begin() + d, _data.end(), subsize._data.begin());

  return subsize;
}

TensorShape TensorShape::transpose(int dim0, int dim1) const {
  dim0 = getRealDim(dim0);
  dim1 = getRealDim(dim1);

  TensorShape size = *this;
  Elem dim0_elem = size._data[dim0];
  size._data[dim0] = size._data[dim1];
  size._data[dim1] = dim0_elem;

  return size;
}

int TensorShape::getRealDim(int d) const {
  CHECK(!empty());
  int rank = getDim();
  if (d < 0) {
    d = rank + d;
  }

  CHECK(d >= 0 && d < rank);
  return d;
}

int TensorShape::getRealIndex(int dim, int index) const {
  CHECK(!empty());
  dim = getRealDim(dim);

  int shape = _data[dim].shape;
  index = index >= 0 ? index : shape + index;

  CHECK(index >= 0 && index <= shape);
  return index;
}

int TensorShape::getDim() const {
  return static_cast<int>(_data.size());
}

bool TensorShape::empty() const {
  return _data.empty();
}

int TensorShape::getShape(int d) const {
  return _data[getRealDim(d)].shape;
}

int TensorShape::getStride(int d) const {
  return _data[getRealDim(d)].stride;
}

int64_t TensorShape::getNumEl() const {
  if (empty()) {
    return 0;
  }
  
  int64_t n = 1;
  for (const Elem &elem : _data) {
    n *= elem.shape;
  }
  return n;
}

void TensorShape::setShape(int dim, ShapeType shape) {
  dim = getRealDim(dim);
  CHECK(dim >= 0 && dim <= this->getDim());
  CHECK(shape <= _data[dim].shape);

  _data[dim].shape = shape;
}

// -- class TensorData ---------------------------------------------------------

TensorData::TensorData(int64_t numel, DType dtype) : _numel(numel), _dtype(dtype) {
  int64_t size_in_bytes = numel * getDTypeSize(dtype);
  _data = new ByteType[size_in_bytes];
}

TensorData::~TensorData() {
  delete[] _data;
  _numel = 0;
  _dtype = DType::kUnknown;
}

// -- class Tensor -------------------------------------------------------------


template<typename T>
Tensor Tensor::create(std::initializer_list<int> shape, util::Span<const T> data) {
  Tensor tensor;

  tensor._shape = TensorShape(shape);
  int64_t numel = tensor._shape.getNumEl();

  DType dtype = getTypeID<T>();
  tensor._data = std::make_shared<TensorData>(numel, dtype);
  tensor._dataPtr = tensor._data->getData();

  // fill data
  CHECK(numel == data.size()) << "data size and shape mismatch";
  std::copy(data.begin(), data.end(), tensor.getData<T>());

  return tensor;
}

template Tensor Tensor::create(std::initializer_list<int> shape, util::Span<const float> data);
template Tensor Tensor::create(std::initializer_list<int> shape, util::Span<const LongType> data);

Tensor::Tensor() : _dataPtr(nullptr) {}
Tensor::~Tensor() {
  _dataPtr = nullptr;
}

Tensor::Tensor(const Tensor &tensor) {
  _data = tensor._data;
  _shape = tensor._shape;
  _dataPtr = tensor._dataPtr;
}

Tensor &Tensor::operator=(const Tensor &tensor) {
  _data = tensor._data;
  _shape = tensor._shape;
  _dataPtr = tensor._dataPtr;

  return *this;
}

Tensor::Tensor(Tensor &&tensor) noexcept {
  _data = tensor._data;
  _shape = std::move(tensor._shape);
  _dataPtr = tensor._dataPtr;
}

Tensor &Tensor::operator=(Tensor &&tensor) {
  _data = tensor._data;
  _shape = std::move(tensor._shape);
  _dataPtr = tensor._dataPtr;

  return *this;
}

// Tensor format
//   byte[4]: "TNSR"
//   int16_t: rank.
//   int16_t: dtype.
//   int32_t * rank: shape.
//   ByteType * SizeOfDType(dtype) * numel: data
//   int16_t: 0x55aa magic number
void Tensor::read(ReadableFile *fp) {
  std::string s = fp->readString(4);
  if (s != "TNSR") {
    throw AbortedException("bad tensor format");
  }

  // rank
  int16_t rank = fp->readValue<int16_t>();
  if (rank > 16 || rank < 0) {
    throw AbortedException("invalid rank");
  }

  // dtype
  int16_t dtype_int16 = fp->readValue<int16_t>();
  DType dtype = static_cast<DType>(dtype_int16);
  if (!IsValidDType(dtype)) {
    throw AbortedException("invalid dtype");
  }

  // shape
  int64_t numel = 1;
  
  std::vector<ShapeType> shape;
  for (int16_t d = 0; d < rank; ++d) {
    int32_t dimension = fp->readValue<int32_t>();
    if (dimension > 65536) {
      throw AbortedException("dimension too big");
    }
    numel *= dimension;
    shape.push_back(dimension);
  }
  if (numel > 1073741824) {
    throw AbortedException("tensor too big");
  }
  _shape = TensorShape(util::makeConstSpan(shape));

  // data
  _data = std::make_shared<TensorData>(numel, dtype);
  util::Span<ByteType> bs_data(_data->getData(), numel);
  fp->readSpan(util::makeSpan(_data->getData(), _data->getSizeInBytes()));
  _dataPtr = _data->getData();

  // magic number
  int16_t magic_number = fp->readValue<int16_t>();
  if (magic_number != 0x55aa) {
    throw AbortedException("invalid magic number");
  }
}

ByteType *Tensor::getRawData(DType dtype) const {
  CHECK(_data);
  CHECK(_data->getDType() == dtype);
  return _dataPtr;
}

Tensor Tensor::view(std::initializer_list<int> shape) const {
  CHECK(isContiguous()) << "only contiguous tensor supports View()";

  std::vector<ShapeType> realShape{shape.begin(), shape.end()};
  std::vector<ShapeType>::iterator itInferred = realShape.end();
  std::vector<ShapeType>::iterator it = realShape.begin();
  int64_t numel = 1;
  for (; it != realShape.end(); ++it) {
    if (*it < 0) {
      CHECK(itInferred == realShape.end()) << "more than 1 inferred dim";
      itInferred = it;
    } else {
      numel *= *it;
    }
  }

  // handle -1 shape
  if (itInferred == realShape.end()) {
    CHECK(numel == this->getNumEl()) << "numel mismatch after View()";
  } else {
    CHECK(this->getNumEl() % numel == 0) << "inferred shape is not a integer";
    *itInferred = static_cast<ShapeType>(this->getNumEl() / numel);
  }


  Tensor view;
  view._data = _data;
  view._dataPtr = _dataPtr;
  view._shape = TensorShape(util::makeConstSpan(realShape));

  CHECK(view.getNumEl() == this->getNumEl());
  return view;
}

bool Tensor::isContiguous() const {
  int numel = 1;
  for (int i = getDim() - 1; i >= 0; --i) {
    if (numel != getStride(i)) return false;
    numel *= getShape(i);
  }

  return true;
}

Tensor Tensor::slice(int dim, int begin, int end) const {
  dim = _shape.getRealDim(dim);
  CHECK(dim >= 0 && dim < this->getDim());

  begin = _shape.getRealIndex(dim, begin);
  end = _shape.getRealIndex(dim, end);
  CHECK(begin >= 0 && begin < end && end <= getShape(dim));

  Tensor tensor;
  tensor._data = _data;
  tensor._shape = _shape;
  tensor._shape.setShape(dim, end - begin);

  int dtypeSize = getDTypeSize(tensor.getDType());
  tensor._dataPtr = _dataPtr + _shape.getStride(dim) * dtypeSize * begin;

  return tensor;
}

Tensor Tensor::slice(int begin, int end) const {
  return slice(0, begin, end);
}

Tensor Tensor::subtensor(int index) const {
  index = _shape.getRealIndex(0, index);
  CHECK(index >= 0 && index < getShape(0));

  Tensor tensor;
  tensor._data = _data;
  tensor._shape = _shape.subsize(1);

  int dtype_size = getDTypeSize(tensor.getDType());
  tensor._dataPtr = _dataPtr + _shape.getStride(0) * dtype_size * index;

  return tensor;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
  Tensor tensor;
  tensor._data = _data;
  tensor._dataPtr = _dataPtr;
  tensor._shape = _shape.transpose(dim0, dim1);

  return tensor;
}

void Tensor::throwIfInvalidShape(std::initializer_list<int> shape) {
  if (shape.size() != getDim()) {
    throw AbortedException(str::sprintf(
        "invalid shape. dim=%d expected, but %d got.", shape.size(), getDim()));
  }

  int i = 0;
  bool correct = true;
  for (int s : shape) {
    if (this->getShape(i) != s) {
      correct = false;
    }
    ++i;
  }

  if (!correct) {
    std::ostringstream actual;
    actual << "(";
    for (int i = 0; i < getDim(); ++i) {
      if (i) actual << ", ";
      actual << this->getShape(i);
    }
    actual << ")";

    std::ostringstream expected;
    bool first = true;
    expected << "(";
    for (int s : shape) {
      if (!first) expected << ", ";
      expected << s;
      first = false;
    }
    expected << ")";

    throw AbortedException(str::sprintf(
        "invalid shape: %s expected, but %s found.", expected.str(), actual.str()));
  }
}

}  // namespace nn
}  // namespace llama
