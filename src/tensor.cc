#include "nn.h"

#include <stdlib.h>
#include <limits>
#include "tensor.h"

namespace llama {
namespace nn {

// ---------------------------------------------------------------------------+
// class DType                                                                |
// ---------------------------------------------------------------------------+

template <>
DType TypeID<float>() {
  return DType::kFloat;
}
template <>
DType TypeID<int64_t>() {
  return DType::kLong;
}

template DType TypeID<float>();
template DType TypeID<int64_t>();

int SizeOfDType(DType dtype) {
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

// ---------------------------------------------------------------------------+
// class Size                                                                 |
// ---------------------------------------------------------------------------+

Size::Size(const Size &size) : data_(size.data_.Copy()) {}
Size::Size(Size &&size) noexcept : data_(std::move(size.data_)) {}
Size &Size::operator=(const Size &size) {
  data_ = size.data_.Copy();
  return *this;
}
Size &Size::operator=(Size &&size) noexcept {
  data_ = std::move(size.data_);
  return *this;
}

Size::Size(util::Span<const ShapeType> shape) {
  data_ = util::FixedArray<Elem>(shape.size());
  util::FixedArray<Elem>::iterator it = data_.begin();
  for (int n : shape) {
    it->shape = n;
    ++it;
  }

  int64_t stride = 1;
  for (int d = shape.size() - 1; d >= 0; --d) {
    CHECK(stride < std::numeric_limits<ShapeType>::max());
    data_[d].stride = static_cast<ShapeType>(stride);
    stride *= data_[d].shape;
  }
}

Size Size::Subsize(int d) const {
  CHECK(d < dim());

  Size subsize;
  subsize.data_ = util::FixedArray<Elem>(dim() - d);
  std::copy(data_.begin() + d, data_.end(), subsize.data_.begin());

  return subsize;
}

Size Size::Transpose(int dim0, int dim1) const {
  dim0 = real_dim(dim0);
  dim1 = real_dim(dim1);

  Size size = *this;
  Elem dim0_elem = size.data_[dim0];
  size.data_[dim0] = size.data_[dim1];
  size.data_[dim1] = dim0_elem;

  return size;
}

int Size::real_dim(int d) const {
  CHECK(!empty());
  int rank = dim();
  if (d < 0) {
    d = rank + d;
  }

  CHECK(d >= 0 && d < rank);
  return d;
}

int Size::real_index(int dim, int index) const {
  CHECK(!empty());
  dim = real_dim(dim);

  int shape = data_[dim].shape;
  index = index >= 0 ? index : shape + index;

  CHECK(index >= 0 && index <= shape);
  return index;
}

int Size::dim() const {
  return static_cast<int>(data_.size());
}

inline bool Size::empty() const {
  return data_.empty();
}

int Size::shape(int d) const {
  return data_[real_dim(d)].shape;
}

int Size::stride(int d) const {
  return data_[real_dim(d)].stride;
}

int64_t Size::numel() const {
  if (empty()) {
    return 0;
  }
  
  int64_t n = 1;
  for (const Elem &elem : data_) {
    n *= elem.shape;
  }
  return n;
}

void Size::set_shape(int dim, ShapeType shape) {
  dim = real_dim(dim);
  CHECK(dim >= 0 && dim <= this->dim());
  CHECK(shape <= data_[dim].shape);

  data_[dim].shape = shape;
}

// ---------------------------------------------------------------------------+
// class TensorData                                                           |
// ---------------------------------------------------------------------------+

TensorData::TensorData(int64_t numel, DType dtype)
    : numel_(numel),
      dtype_(dtype) {
  int64_t size_in_bytes = numel * SizeOfDType(dtype);
  data_ = new ByteType[size_in_bytes];
}

TensorData::~TensorData() {
  delete[] data_;
  numel_ = 0;
  dtype_ = DType::kUnknown;
}

// ---------------------------------------------------------------------------+
// class Tensor                                                               |
// ---------------------------------------------------------------------------+

Tensor::Tensor() : data_ptr_(nullptr) {}
Tensor::~Tensor() {
  data_ptr_ = nullptr;
}

Tensor::Tensor(const Tensor &tensor) {
  data_ = tensor.data_;
  size_ = tensor.size_;
  data_ptr_ = tensor.data_ptr_;
}

Tensor &Tensor::operator=(const Tensor &tensor) {
  data_ = tensor.data_;
  size_ = tensor.size_;
  data_ptr_ = tensor.data_ptr_;

  return *this;
}

Tensor::Tensor(Tensor &&tensor) noexcept {
  data_ = tensor.data_;
  size_ = std::move(tensor.size_);
  data_ptr_ = tensor.data_ptr_;
}

Tensor &Tensor::operator=(Tensor &&tensor) {
  data_ = tensor.data_;
  size_ = std::move(tensor.size_);
  data_ptr_ = tensor.data_ptr_;

  return *this;
}

// Tensor format
//   byte[4]: "TNSR"
//   int16_t: rank.
//   int16_t: dtype.
//   int32_t * rank: shape.
//   ByteType * SizeOfDType(dtype) * numel: data
//   int16_t: 0x55aa magic number
Status Tensor::Read(ReadableFile *fp) {
  std::string s;
  RETURN_IF_ERROR(fp->ReadString(4, &s));
  if (s != "TNSR") {
    RETURN_ABORTED() << "bad tensor format";
  }

  // rank
  int16_t rank;
  RETURN_IF_ERROR(fp->ReadValue(&rank));
  if (rank > 16 || rank < 0) {
    RETURN_ABORTED() << "invalid rank";
  }

  // dtype
  int16_t dtype_int16;
  RETURN_IF_ERROR(fp->ReadValue(&dtype_int16));
  DType dtype = static_cast<DType>(dtype_int16);
  if (!IsValidDType(dtype)) {
    RETURN_ABORTED() << "invalid dtype";
  }

  // shape
  int64_t numel = 1;
  int32_t dimension;
  std::vector<ShapeType> shape;
  for (int16_t d = 0; d < rank; ++d) {
    RETURN_IF_ERROR(fp->ReadValue(&dimension));
    if (dimension > 65536) {
      RETURN_ABORTED() << "dimension too big";
    }
    numel *= dimension;
    shape.push_back(dimension);
  }
  if (numel > 4194304) {
    RETURN_ABORTED() << "tensor too big";
  }
  size_ = Size(util::MakeConstSpan(shape));

  // data
  data_ = std::make_shared<TensorData>(numel, dtype);
  util::Span<ByteType> bs_data(data_->data(), numel);
  RETURN_IF_ERROR(fp->ReadSpan(util::MakeSpan(
      data_->data(),
      data_->size_in_bytes())));
  data_ptr_ = data_->data();

  // magic number
  int16_t magic_number;
  RETURN_IF_ERROR(fp->ReadValue(&magic_number));
  if (magic_number != 0x55aa) {
    RETURN_ABORTED() << "invalid magic number";
  }

  return OkStatus();
}

ByteType *Tensor::raw_data(DType dtype) const {
  CHECK(data_);
  CHECK(data_->dtype() == dtype);
  return data_ptr_;
}

Tensor Tensor::View(std::initializer_list<int> shape) const {
  CHECK(is_contiguous()) << "only contiguous tensor supports View()";

  std::vector<ShapeType> real_shape{shape.begin(), shape.end()};
  std::vector<ShapeType>::iterator it_inferred = real_shape.end();
  std::vector<ShapeType>::iterator it = real_shape.begin();
  int64_t numel = 1;
  for (; it != real_shape.end(); ++it) {
    if (*it < 0) {
      CHECK(it_inferred == real_shape.end()) << "more than 1 inferred dim";
      it_inferred = it;
    } else {
      numel *= *it;
    }
  }

  // handle -1 shape
  if (it_inferred == real_shape.end()) {
    CHECK(numel == this->numel()) << "numel mismatch after View()";
  } else {
    CHECK(this->numel() % numel == 0) << "inferred shape is not a integer";
    *it_inferred = this->numel() / numel;
  }


  Tensor view;
  view.data_ = data_;
  view.data_ptr_ = data_ptr_;
  view.size_ = Size(util::MakeConstSpan(real_shape));

  CHECK(view.numel() == this->numel());
  return view;
}

bool Tensor::is_contiguous() const {
  int numel = 1;
  for (int i = dim() - 1; i >= 0; --i) {
    if (numel != stride(i)) return false;
    numel *= shape(i);
  }

  return true;
}

Tensor Tensor::Slice(int dim, int begin, int end) const {
  dim = size_.real_dim(dim);
  CHECK(dim >= 0 && dim < this->dim());

  begin = size_.real_index(dim, begin);
  end = size_.real_index(dim, end);
  CHECK(begin >= 0 && begin < end && end <= shape(dim));

  Tensor tensor;
  tensor.data_ = data_;
  tensor.size_ = size_;
  tensor.size_.set_shape(dim, end - begin);

  int dtype_size = SizeOfDType(tensor.dtype());
  tensor.data_ptr_ = data_ptr_ + size_.stride(dim) * dtype_size * begin;

  return tensor;
}

Tensor Tensor::Slice(int begin, int end) const {
  return Slice(0, begin, end);
}

Tensor Tensor::Subtensor(int index) const {
  index = size_.real_index(0, index);
  CHECK(index >= 0 && index < shape(0));

  Tensor tensor;
  tensor.data_ = data_;
  tensor.size_ = size_.Subsize(1);

  int dtype_size = SizeOfDType(tensor.dtype());
  tensor.data_ptr_ = data_ptr_ + size_.stride(0) * dtype_size * index;

  return tensor;
}

Tensor Tensor::Transpose(int dim0, int dim1) const {
  Tensor tensor;
  tensor.data_ = data_;
  tensor.data_ptr_ = data_ptr_;
  tensor.size_ = size_.Transpose(dim0, dim1);

  return tensor;
}

}  // namespace nn
}  // namespace llama
