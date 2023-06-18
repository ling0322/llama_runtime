#include "nn.h"

#include <stdlib.h>
#include <limits>
#include "llmrt_blas.h"
#include "operators.h"

namespace llama {
namespace nn {

// -- class Device ----------

Device::Device() : _type(Type::kUnknown) {}
Device::Device(Type type) : _type(type) {}

Device Device::createForCPU() {
  return Device(Type::kCpu);
}

// -- class TensorMap ----------

// tensor_dict format
//   byte[4]: "TDIC"
//   int32_t: num_record
//   Record[num_record]:
//     int16_t: name_len
//     byte[name_len]: name
//     Tensor
//   int16_t: magic number 0x55aa
void TensorMap::read(const std::string &filename) {
  _dict.clear();

  auto fp = ReadableFile::open(filename);

  std::string s = fp->readString(4);
  if (s != "TDIC") {
    throw AbortedException("invalid tensor_dict file");
  }

  int32_t numRecord = fp->readValue<int32_t>();
  for (int i = 0; i < numRecord; ++i) {
    int16_t nameLen = fp->readValue<int16_t>();
    if (nameLen <= 0) {
      throw AbortedException("invalid tensor_dict file (name_len)");
    }

    Tensor tensor;
    std::string name = fp->readString(nameLen);
    tensor.read(fp.get());
    _dict[name] = std::move(tensor);
  }

  // magic number
  int16_t magicNumber = fp->readValue<int16_t>();
  if (magicNumber != 0x55aa) {
    throw AbortedException("invalid tensor_dict file (magic number)");
  }
}

Tensor TensorMap::getTensor(const std::string &name) const {
  auto it = _dict.find(name);
  CHECK(it != _dict.end());

  return it->second;
}

bool TensorMap::getTensorNoThrow(const std::string &name, Tensor *tensor) const {
  auto it = _dict.find(name);
  if (it == _dict.end()) {
    return false;
  }

  *tensor = it->second;
  return true;
}

void TensorMap::putTensor(const std::string &name, TensorCRef tensor) {
  _dict[name] = tensor;
}

bool TensorMap::hasTensor(const std::string &name) const {
  return _dict.find(name) != _dict.end();
}

// -- class Context ----------

Context::Context() : _F(nullptr) {}

Context Context::withName(const std::string &name) const {
  CHECK(!name.empty());
  Context ctx;
  ctx._F = _F;
  ctx._device = _device;
  ctx._ns = this->name(name);

  return ctx;
}

std::string Context::name(const std::string &name) const {
  std::string ns = _ns;

  if (ns.empty()) {
    ns = name;
  } else {
    ns += ".";
    ns += name;
  }

  return ns;
}

// -- class Linear ------------------------------------------------------------

Linear::Linear() : _inFeatures(0), _outFeatures(0) {}

std::unique_ptr<Linear> Linear::create(const Context &ctx, int inFeatures, int outFeatures) {
  std::unique_ptr<Linear> linear{new Linear()};
  if (inFeatures <= 0 || outFeatures <= 0) {
    throw AbortedException("invalid d_model");
  }

  linear->_inFeatures = inFeatures;
  linear->_outFeatures = outFeatures;
  linear->_ctx = ctx;
  return linear;
}

void Linear::initParameters(const TensorMap &stateDict) {
  std::string nameW = _ctx.name(kWeight);
  std::string nameB = _ctx.name(kBias);

  _w = stateDict.getTensor(nameW);
  _b = stateDict.getTensor(nameB);

  _w.throwIfInvalidShape({_outFeatures, _inFeatures});
  _b.throwIfInvalidShape({_outFeatures});
}

Tensor Linear::forward(const Tensor &input) const {
  Operators *F = _ctx.F();
  Tensor x;
  if (input.getDim() == 1) {
    x = F->gemv(_w, input);
  } else if (input.getDim() == 2) {
    x = F->gemm(input, _w.transpose(0, 1));
  } else if (input.getDim() > 2) {
    x = F->bmm(input, _w.transpose(0, 1));
  }
  x = F->add(x, _b);

  return x;
}

// -- class LayerNorm ---------------------------------------------------------

LayerNorm::LayerNorm() : _dModel(0), _eps(0.0f) {}

std::unique_ptr<LayerNorm> LayerNorm::create(const Context &ctx, int dModel, float eps) {
  std::unique_ptr<LayerNorm> layer{new LayerNorm()};
  if (dModel <= 0 || eps <= 0.0f) {
    throw AbortedException("invalid dModel or eps");
  }

  layer->_dModel = dModel;
  layer->_eps = eps;
  layer->_ctx = ctx;
  return layer;
}

void LayerNorm::initParameters(const TensorMap &stateDict) {
  std::string nameW = _ctx.name(kWeight);
  std::string nameB = _ctx.name(kBias);

  _w = stateDict.getTensor(nameW);
  _b = stateDict.getTensor(nameB);

  _w.throwIfInvalidShape({_dModel});
  _b.throwIfInvalidShape({_dModel});
}

Tensor LayerNorm::forward(const Tensor &input) const {
  Operators *F = _ctx.F();
  return F->layerNorm(input, _w, _b, _eps);
}

}  // namespace nn
}  // namespace llama
