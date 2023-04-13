#include "model.h"

#include "log.h"

namespace llama {
namespace model {

// ----------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------

const Port *Model::get_input(const std::string &name) const noexcept {
  for (const Port &port : Inputs()) {
    if (name == port.name())
      return &port;
  }
  return nullptr;
}

const Port *Model::get_output(const std::string& name) const noexcept {
  for (const Port &port : Outputs()) {
    if (name == port.name()) 
      return &port;
  }
  return nullptr;
}

const Port *Model::get_input(int index) const noexcept {
  if (index >= 0 && index < Inputs().size()) {
    return &(Inputs()[index]);
  } else {
    return nullptr;
  }
}

const Port *Model::get_output(int index) const noexcept {
  if (index >= 0 && index < Outputs().size()) {
    return &(Outputs()[index]);
  } else {
    return nullptr;
  }
}

int Model::num_inputs() const noexcept {
  return Inputs().size();
}
int Model::num_outputs() const noexcept {
  return Outputs().size();
}

// ----------------------------------------------------------------------------
// InferRequest
// ----------------------------------------------------------------------------

void InferRequest::SetInput(PCStrType name, const TensorView &tensor) {
  Value borrowed_value = CreateValue(tensor);

  borrowed_values_.emplace_back(std::move(borrowed_value));
  SetInput(name, borrowed_values_.back());
}

// ----------------------------------------------------------------------------
// Value
// ----------------------------------------------------------------------------

Value::Value(std::unique_ptr<ImplBase> &&impl) : impl_(std::move(impl)) {}

Status Value::GetTensor(TensorView *tensor) {
  LL_CHECK(impl_);
  return impl_->GetTensor(tensor);
}

// ----------------------------------------------------------------------------
// Port
// ----------------------------------------------------------------------------

Port::Port(): type_(Type::kUnknown), dtype_(DType::kUnknown) {}

Port::Type Port::type() const {
    return type_;
}
void Port::set_type(Type type) {
    type_ = type_;
}
PCStrType Port::name() const {
  return name_.c_str();
}
void Port::set_name(const std::string &name) {
  name_ = name;
}
DType Port::dtype() const {
  return dtype_;
}
void Port::set_dtype(DType dtype) {
  dtype_ = dtype;
}
int Port::rank() const {
  return static_cast<int>(shape_.size());
}
int Port::shape(int d) const {
  return shape_[d];
}
void Port::set_shape(const std::vector<int> &shape) {
  shape_ = shape;
}

}  // namespace model
}  // namespace llama
