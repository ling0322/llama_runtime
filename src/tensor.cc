#include "nn.h"

#include <stdlib.h>
#include <limits>
#include "tensor.h"

namespace llama {
namespace nn {

TensorData::TensorData(int numel, DType dtype)
    : numel_(numel),
      dtype_(dtype) {
  int size_in_bytes = numel * SizeOfDType(dtype);
  data_ = new ByteType[size_in_bytes];
}

TensorData::~TensorData() {
  delete[] data_;
  numel_ = 0;
  dtype_ = DType::kUnknown;
}

Tensor::Tensor() : data_ptr_(nullptr) {}
Tensor::~Tensor() {
  data_ptr_ = nullptr;
}

Tensor::Tensor(const Tensor &tensor) {
  data_ = tensor.data_;
  shape_ = tensor.shape_.Copy();
  data_ptr_ = tensor.data_ptr_;
}

Tensor &Tensor::operator=(const Tensor &tensor) {
  data_ = tensor.data_;
  shape_ = tensor.shape_.Copy();
  data_ptr_ = tensor.data_ptr_;

  return *this;
}

Tensor::Tensor(Tensor &&tensor) noexcept {
  data_ = tensor.data_;
  shape_ = std::move(tensor.shape_);
  data_ptr_ = tensor.data_ptr_;
}

Tensor &Tensor::operator=(Tensor &&tensor) {
  data_ = tensor.data_;
  shape_ = std::move(tensor.shape_);
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
  DType dtype = static_cast<DType>(dtype);
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
  }
  if (numel > 4194304) {
    RETURN_ABORTED() << "tensor too big";
  }
  FillShapeStride(util::MakeConstSpan(shape));

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

int Tensor::real_dim(int d) const {
  CHECK(!empty());
  int _rank = rank();
  if (d < 0) {
    d = _rank + d;
  }

  CHECK(d >= 0 && d < _rank);
  return d;
}

int Tensor::shape(int d) const {
  return shape_[real_dim(d)].dimension;
}

int Tensor::stride(int d) const {
  return shape_[real_dim(d)].stride;
}

int Tensor::numel() const {
  if (empty()) {
    return 0;
  }
  
  int n = 1;
  for (const Shape &shape : shape_) {
    n *= shape.dimension;
  }
  return n;
}

ByteType *Tensor::raw_data(DType dtype) const {
  CHECK(data_);
  CHECK(data_->dtype() == dtype);
  return data_ptr_;
}

void Tensor::FillShapeStride(util::Span<const int> shape) {
  shape_ = util::FixedArray<Shape>(shape.size());
  Shape *ps = shape_.data();
  for (int dimension : shape) {
    ps->dimension = dimension;
    ++ps;
  }

  int64_t stride = 1;
  for (int d = shape.size() - 1; d >= 0; --d) {
    CHECK(stride < std::numeric_limits<ShapeType>::max());
    shape_[d].stride = stride;
    stride *= shape_[d].dimension;
  }
}

Tensor Tensor::View(std::initializer_list<int> shape) const {
  Tensor view;
  view.data_ = data_;
  view.data_ptr_ = data_ptr_;
  view.FillShapeStride(util::MakeConstSpan(shape));

  CHECK(view.numel() == numel());
  return view;
}

Tensor Tensor::Transpose(int dim0, int dim1) const {
  dim0 = real_dim(dim0);
  dim1 = real_dim(dim1);

  Tensor tensor;
  tensor.data_ = data_;
  tensor.data_ptr_ = data_ptr_;
  tensor.shape_ = shape_.Copy();

  Shape dim0_shape = tensor.shape_[dim0];
  tensor.shape_[dim0] = tensor.shape_[dim1];
  tensor.shape_[dim1] = dim0_shape;

  return tensor;
}

}  // namespace nn
}  // namespace llama
