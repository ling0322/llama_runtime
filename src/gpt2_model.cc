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
    : n_embd(0),
      n_ctx(0),
      vocab_size(0) {
}

StatusOr<GPT2Config> GPT2Config::FromIni(const IniParser &ini) {
  auto config = std::make_unique<GPT2Config>();

  RETURN_IF_ERROR(ini.Get(kConfigSection, "n_embd", &config->n_embd));
  RETURN_IF_ERROR(ini.Get(kConfigSection, "n_ctx", &config->n_ctx));
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

  return model;
}

Status GPT2Model::InitParameters(const TensorMap &state_dict) {
  Tensor wte;
  RETURN_IF_ERROR(state_dict.TryGet(kWte, &wte));
  if (wte.dim() != 2 || wte.shape(0) != config_.vocab_size ||
      wte.shape(1) != config_.n_embd) {
    RETURN_ABORTED() << "invalid tensor: wte";
  }
  wte_ = wte;

  Tensor wpe;
  RETURN_IF_ERROR(state_dict.TryGet(kWpe, &wpe));
  if (wpe.dim() != 2 || wpe.shape(0) != config_.n_ctx ||
      wte.shape(1) != config_.n_embd) {
    RETURN_ABORTED() << "invalid tensor: wpe";
  }
  wpe_ = wpe;

  return OkStatus();
}

Tensor GPT2Model::Forward(TensorMap *past, TensorCRef input) const {
  CHECK(input.dim() == 2 && input.dtype() == DType::kLong);

  Operators *F = ctx_.F();
  Tensor x = F->Lookup(wte_, input);

  Tensor pos_emb = wpe_.Slice(0, x.shape(1));
  x = F->Add(x, pos_emb);

  return x;
}

}  // namespace nn
}  // namespace llama
