#include "nn.h"

#include <stdlib.h>
#include <limits>
#include "gemm.h"
#include "operators.h"
#include "status.h"

namespace llama {
namespace nn {

// ---------------------------------------------------------------------------+
// class Module                                                               |
// ---------------------------------------------------------------------------+

Tensor Module::Forward(const Tensor &input) const {
  NOT_IMPL();
  return Tensor();
}

Tensor Module::Forward(TensorDict *cache, const Tensor &input) const {
  NOT_IMPL();
  return Tensor();
}

// ---------------------------------------------------------------------------+
// class Device                                                               |
// ---------------------------------------------------------------------------+

Device::Device() : type_(Type::kUnknown) {}
Device::Device(Type type) : type_(type) {}

Device Device::CPU() {
  return Device(Type::kCpu);
}

// ---------------------------------------------------------------------------+
// class TensorDict                                                           |
// ---------------------------------------------------------------------------+

// tensor_dict format
//   byte[4]: "TDIC"
//   int32_t: num_record
//   Record[num_record]:
//     int16_t: name_len
//     byte[name_len]: name
//     Tensor
//   int16_t: magic number 0x55aa
Status TensorDict::Read(const std::string &filename) {
  dict_.clear();

  StatusOr<ReadableFile> fp = ReadableFile::Open(filename);
  RETURN_IF_ERROR(fp);

  std::string s;
  RETURN_IF_ERROR(fp->ReadString(4, &s));
  if (s != "TDIC") {
    RETURN_ABORTED() << "invalid tensor_dict file";
  }

  int32_t num_record;
  RETURN_IF_ERROR(fp->ReadValue(&num_record));
  for (int i = 0; i < num_record; ++i) {
    int16_t name_len;
    RETURN_IF_ERROR(fp->ReadValue(&name_len));
    if (name_len <= 0) {
      RETURN_ABORTED() << "invalid tensor_dict file (name_len)";
    }

    std::string name;
    Tensor tensor;
    RETURN_IF_ERROR(fp->ReadString(name_len, &name));
    RETURN_IF_ERROR(tensor.Read(fp.get()));
    dict_[name] = std::move(tensor);
  }

  // magic number
  int16_t magic_number;
  RETURN_IF_ERROR(fp->ReadValue(&magic_number));
  if (magic_number != 0x55aa) {
    RETURN_ABORTED() << "invalid tensor_dict file (magic number)";
  }

  return OkStatus();
}

Tensor &TensorDict::operator[](const std::string &name) {
  auto it = dict_.find(name);
  CHECK(it != dict_.end());

  return it->second;
}

Status TensorDict::get(const std::string &name, Tensor *tensor) const {
  auto it = dict_.find(name);
  if (it == dict_.end()) {
    RETURN_ABORTED() << "tensor " << name << " not found";
  }

  *tensor = it->second;
  return OkStatus();
}

bool TensorDict::has_tensor(const std::string &name) const {
  return dict_.find(name) != dict_.end();
}

// ---------------------------------------------------------------------------+
// class Context                                                              |
// ---------------------------------------------------------------------------+

Context::Context() : F_(nullptr) {}

Context Context::WithName(const std::string &name) const {
  CHECK(!name.empty());
  Context ctx;
  ctx.F_ = F_;
  ctx.device_ = device_;
  ctx.ns_ = this->name(name);

  return ctx;
}

std::string Context::name(const std::string &name) const {
  std::string ns = ns_;

  if (ns.empty()) {
    ns = name;
  } else {
    ns += ".";
    ns += name;
  }

  return ns;
}

// ---------------------------------------------------------------------------+
// class Linear                                                               |
// ---------------------------------------------------------------------------+

Linear::Linear() : in_features_(0), out_features_(0) {}

StatusOr<Linear> Linear::Create(
    const Context &ctx,
    int in_features,
    int out_features) {
  std::unique_ptr<Linear> linear{new Linear()};
  if (in_features <= 0 || out_features <= 0) {
    RETURN_ABORTED() << "invalid d_model";
  }

  linear->in_features_ = in_features;
  linear->out_features_ = out_features;
  linear->ctx_ = ctx;
  return linear;
}

Status Linear::InitParameters(const TensorDict &state_dict) {
  std::string name_w = ctx_.name(kWeight);
  RETURN_IF_ERROR(state_dict.get(name_w, &w_));
  if (w_.rank() != 2 || w_.shape(0) != out_features_ ||
      w_.shape(1) != in_features_) {
    RETURN_ABORTED() << "invalid shape of tensor " << name_w;
  }

  std::string name_b = ctx_.name(kBias);
  RETURN_IF_ERROR(state_dict.get(name_b, &b_));
  if (b_.rank() != 1 || b_.shape(0) != out_features_) {
    RETURN_ABORTED() << "invalid shape of tensor " << name_b;
  }

  return OkStatus();
}

Tensor Linear::Forward(const Tensor &input) const {
  Operators *F = ctx_.F();
  Tensor x = F->MatMul(input, w_.Transpose(0, 1));
  x = F->Add(x, b_);

  return x;
}

// ---------------------------------------------------------------------------+
// class LayerNorm                                                            |
// ---------------------------------------------------------------------------+

LayerNorm::LayerNorm() : d_model_(0), eps_(0.0f) {}

StatusOr<LayerNorm> LayerNorm::Create(
    const Context &ctx,
    int d_model,
    float eps) {
  std::unique_ptr<LayerNorm> layer{new LayerNorm()};
  if (d_model <= 0 || eps <= 0.0f) {
    RETURN_ABORTED() << "invalid d_model or eps";
  }

  layer->d_model_ = d_model;
  layer->eps_ = eps;
  layer->ctx_ = ctx;
  return layer;
}

Status LayerNorm::InitParameters(const TensorDict &state_dict) {
  std::string name_w = ctx_.name(kWeight);
  RETURN_IF_ERROR(state_dict.get(name_w, &w_));
  if (w_.rank() != 1 || w_.shape(0) != d_model_) {
    RETURN_ABORTED() << "invalid shape of tensor " << name_w;
  }

  std::string name_b = ctx_.name(kBias);
  RETURN_IF_ERROR(state_dict.get(name_b, &b_));
  if (b_.rank() != 1 || b_.shape(0) != d_model_) {
    RETURN_ABORTED() << "invalid shape of tensor " << name_b;
  }

  return OkStatus();
}

Tensor LayerNorm::Forward(const Tensor &input) const {
  Operators *F = ctx_.F();
  return F->LayerNorm(input, w_, b_, eps_);
}

}  // namespace nn
}  // namespace llama
