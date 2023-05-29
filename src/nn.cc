#include "nn.h"

#include <stdlib.h>
#include <limits>
#include "gemm.h"
#include "operators.h"
#include "status.h"

namespace llama {
namespace nn {

// -- class Device ------------------------------------------------------------

Device::Device() : type_(Type::kUnknown) {}
Device::Device(Type type) : type_(type) {}

Device Device::CPU() {
  return Device(Type::kCpu);
}

// -- class TensorMap ---------------------------------------------------------

// tensor_dict format
//   byte[4]: "TDIC"
//   int32_t: num_record
//   Record[num_record]:
//     int16_t: name_len
//     byte[name_len]: name
//     Tensor
//   int16_t: magic number 0x55aa
Status TensorMap::Read(const std::string &filename) {
  dict_.clear();

  expected_ptr<ReadableFile> fp = ReadableFile::Open(filename);
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

Tensor TensorMap::Get(const std::string &name) {
  auto it = dict_.find(name);
  CHECK(it != dict_.end());

  return it->second;
}

Status TensorMap::TryGet(const std::string &name, Tensor *tensor) const {
  auto it = dict_.find(name);
  if (it == dict_.end()) {
    RETURN_ABORTED() << "tensor \"" << name << "\" not found.";
  }

  *tensor = it->second;
  return OkStatus();
}

void TensorMap::Put(const std::string &name, TensorCRef tensor) {
  dict_[name] = tensor;
}

bool TensorMap::exists(const std::string &name) const {
  return dict_.find(name) != dict_.end();
}

// -- class Context -----------------------------------------------------------

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

// -- class Linear ------------------------------------------------------------

Linear::Linear() : in_features_(0), out_features_(0) {}

expected_ptr<Linear> Linear::Create(
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

Status Linear::InitParameters(const TensorMap &state_dict) {
  std::string name_w = ctx_.name(kWeight);
  std::string name_b = ctx_.name(kBias);

  RETURN_IF_ERROR(state_dict.TryGet(name_w, &w_));
  RETURN_IF_ERROR(state_dict.TryGet(name_b, &b_));

  RETURN_IF_ERROR(w_.CheckShape({out_features_, in_features_})) << name_w;
  RETURN_IF_ERROR(b_.CheckShape({out_features_})) << name_b;

  return OkStatus();
}

Tensor Linear::Forward(const Tensor &input) const {
  Operators *F = ctx_.F();
  Tensor x = F->MatMul(input, w_.Transpose(0, 1));
  x = F->Add(x, b_);

  return x;
}

// -- class LayerNorm ---------------------------------------------------------

LayerNorm::LayerNorm() : d_model_(0), eps_(0.0f) {}

expected_ptr<LayerNorm> LayerNorm::Create(
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

Status LayerNorm::InitParameters(const TensorMap &state_dict) {
  std::string name_w = ctx_.name(kWeight);
  std::string name_b = ctx_.name(kBias);

  RETURN_IF_ERROR(state_dict.TryGet(name_w, &w_));
  RETURN_IF_ERROR(state_dict.TryGet(name_b, &b_));

  RETURN_IF_ERROR(w_.CheckShape({d_model_})) << name_w;
  RETURN_IF_ERROR(b_.CheckShape({d_model_})) << name_b;

  return OkStatus();
}

Tensor LayerNorm::Forward(const Tensor &input) const {
  Operators *F = ctx_.F();
  return F->LayerNorm(input, w_, b_, eps_);
}

}  // namespace nn
}  // namespace llama
