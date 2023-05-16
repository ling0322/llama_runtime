#include "gpt2_model.h"

#include <memory>
#include "ini_parser.h"
#include "operators.h"

namespace llama {
namespace nn {

// ---------------------------------------------------------------------------+
// class GPT2Config                                                           |
// ---------------------------------------------------------------------------+

GPT2Config::GPT2Config()
    : d_model(0),
      vocab_size(0) {
}

StatusOr<GPT2Config> GPT2Config::FromIni(const IniParser &ini) {
  auto config = std::make_unique<GPT2Config>();

  RETURN_IF_ERROR(ini.Get(kConfigSection, "d_model", &config->d_model));
  RETURN_IF_ERROR(ini.Get(kConfigSection, "vocab_size", &config->vocab_size));

  return config;
}

// ---------------------------------------------------------------------------+
// class GPT2Model                                                            |
// ---------------------------------------------------------------------------+

GPT2Model::GPT2Model() {}

StatusOr<GPT2Model> GPT2Model::Create(const Context &ctx, GPT2Config config) {
  std::unique_ptr<GPT2Model> model{new GPT2Model()};
  
  model->ctx_ = ctx;
  model->config_ = config;
  if (config.d_model <= 0) {
    RETURN_ABORTED() << "invalid d_model";
  }

  return model;
}

Status GPT2Model::InitParameters(const TensorMap &state_dict) {
  Tensor wte;
  
  RETURN_IF_ERROR(state_dict.TryGet(kWte, &wte));
  if (wte.dim() != 2 || wte.shape(0) != config_.vocab_size ||
      wte.shape(1) != config_.d_model) {
    RETURN_ABORTED() << "invalid tensor: wte";
  }
  wte_ = wte;

  return OkStatus();
}

Tensor GPT2Model::Forward(TensorMap *past, TensorCRef input) const {
  Operators *F = ctx_.F();
  Tensor x = F->Lookup(wte_, input);

  return x;
}

}  // namespace nn
}  // namespace llama
