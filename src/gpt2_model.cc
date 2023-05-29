#include "gpt2_model.h"

#include <memory>
#include "ini_config.h"
#include "operators.h"
#include "strings.h"

namespace llama {
namespace nn {

// ---------------------------------------------------------------------------+
// class GPT2Config                                                           |
// ---------------------------------------------------------------------------+

GPT2Config::GPT2Config()
    : n_embd(0),
      n_ctx(0),
      n_inner(0),
      n_head(0),
      n_layer(0),
      vocab_size(0),
      hidden_size(0) {
}

expected_ptr<GPT2Config> GPT2Config::FromIni(const IniConfig &ini) {
  auto config = std::make_unique<GPT2Config>();

  RETURN_IF_ERROR(ini.EnsureSection(kSection));
  const IniSection &section = ini.section(kSection);

  RETURN_IF_ERROR(section.Get("n_embd", &config->n_embd));
  RETURN_IF_ERROR(section.Get("n_ctx", &config->n_ctx));
  RETURN_IF_ERROR(section.Get("n_inner", &config->n_inner));
  RETURN_IF_ERROR(section.Get("n_head", &config->n_head));
  RETURN_IF_ERROR(section.Get("n_layer", &config->n_layer));
  RETURN_IF_ERROR(section.Get("vocab_size", &config->vocab_size));
  RETURN_IF_ERROR(section.Get("hidden_size", &config->hidden_size));

  return config;
}

// ---------------------------------------------------------------------------+
// class GPT2Block                                                            |
// ---------------------------------------------------------------------------+

GPT2Block::GPT2Block() {}

expected_ptr<GPT2Block> GPT2Block::Create(
    const Context &ctx,
    GPT2Config config) {
  std::unique_ptr<GPT2Block> block{new GPT2Block()};
  
  block->ctx_ = ctx;
  block->config_ = config;

  int d_model = config.hidden_size;
  int n_inner = config.n_inner;

  auto ln1 = LayerNorm::Create(ctx.WithName(kLn1), d_model);
  auto ln2 = LayerNorm::Create(ctx.WithName(kLn2), d_model);
  auto fc = Linear::Create(ctx.WithName(kFc), d_model, n_inner);
  auto proj = Linear::Create(ctx.WithName(kProj), n_inner, d_model);
  auto attn = MultiheadSelfAttention::Create(
      ctx.WithName(kAttn),
      config.n_head,
      d_model);

  RETURN_IF_ERROR(ln1);
  RETURN_IF_ERROR(ln2);
  RETURN_IF_ERROR(fc);
  RETURN_IF_ERROR(proj);
  RETURN_IF_ERROR(attn);

  block->ln2_ = std::move(ln2).unique_ptr();
  block->ln1_ = std::move(ln1).unique_ptr();
  block->attn_ = std::move(attn).unique_ptr();
  block->fc_ = std::move(fc).unique_ptr();
  block->proj_ = std::move(proj).unique_ptr();

  return block;
}

Status GPT2Block::InitParameters(const TensorMap &state_dict) {
  RETURN_IF_ERROR(ln2_->InitParameters(state_dict));
  RETURN_IF_ERROR(ln1_->InitParameters(state_dict));
  RETURN_IF_ERROR(attn_->InitParameters(state_dict));
  RETURN_IF_ERROR(fc_->InitParameters(state_dict));
  RETURN_IF_ERROR(proj_->InitParameters(state_dict));

  return OkStatus();
}

Tensor GPT2Block::Forward(TensorMap *past,
                          TensorCRef input,
                          TensorCRef mask) const {
  Operators *F = ctx_.F();

  Tensor residual = input;
  Tensor x = ln1_->Forward(input);
  x = attn_->Forward(past, x, mask);
  x = F->Add(x, residual);

  residual = x;
  x = ln2_->Forward(x);
  x = fc_->Forward(x);
  x = F->GELU(x);
  x = proj_->Forward(x);
  x = F->Add(x, residual);

  return x;
}

// -- class GPT2Model ----------------------------------------------------------

GPT2Model::GPT2Model() {}

expected_ptr<GPT2Model> GPT2Model::Create(
    const Context &ctx,
    GPT2Config config) {
  std::unique_ptr<GPT2Model> model{new GPT2Model()};

  model->ctx_ = ctx.WithName(kGpt2);
  model->config_ = config;

  for (int i = 0; i < config.n_layer; ++i) {
    std::string block_name = strings::Sprintf("%s%d", kBlock, i);
    auto block = GPT2Block::Create(model->ctx_.WithName(block_name), config);
    RETURN_IF_ERROR(block);

    model->blocks_.emplace_back(std::move(block).unique_ptr());
  }

  auto ln = LayerNorm::Create(model->ctx_.WithName(kLnF), config.hidden_size);
  RETURN_IF_ERROR(ln);
  model->ln_f_ = std::move(ln).unique_ptr();

  return model;
}

Status GPT2Model::InitParameters(const TensorMap &state_dict) {
  Operators *F = ctx_.F();

  int vocab_size = config_.vocab_size;
  int n_embd = config_.n_embd;
  int n_ctx = config_.n_ctx;

  RETURN_IF_ERROR(state_dict.TryGet(ctx_.name(kWte), &wte_));
  RETURN_IF_ERROR(state_dict.TryGet(ctx_.name(kWpe), &wpe_));

  RETURN_IF_ERROR(wte_.CheckShape({vocab_size, n_embd})) << ctx_.name(kWte);
  RETURN_IF_ERROR(wpe_.CheckShape({n_ctx, n_embd})) << ctx_.name(kWpe);

  for (const std::unique_ptr<GPT2Block> &block : blocks_) {
    RETURN_IF_ERROR(block->InitParameters(state_dict));
  }
  RETURN_IF_ERROR(ln_f_->InitParameters(state_dict));

  mask_ = F->CausalMask(config_.n_ctx);
  return OkStatus();
}

Tensor GPT2Model::Forward(TensorMap *past, TensorCRef input_ids) const {
  CHECK(input_ids.dim() == 2 && input_ids.dtype() == DType::kLong);

  Operators *F = ctx_.F();
  Tensor x = F->Lookup(wte_, input_ids);

  // seq_len is the number of token ids processed so far. Which is used to
  // get the correct index of positional embedding for current `input_ids`.
  // Since `past` could only accept tensors, we use a scalar Tensor to store
  // it.
  int input_start_idx = 0;
  if (past) {
    std::string name = ctx_.name(kSeqLen);
    if (past->exists(name)) {
      Tensor x = past->Get(name);
      input_start_idx = static_cast<int>(x.elem<LongType>({0}));
    }

    int next_idx = input_start_idx + input_ids.shape(1);
    past->Put(name, Tensor::FromData<LongType>({1}, {next_idx}));
  }
  Tensor pos_emb = wpe_.Slice(input_start_idx, input_start_idx + x.shape(1));
  x = F->Add(x, pos_emb);

  int i = 0;
  for (const std::unique_ptr<GPT2Block> &block : blocks_) {
    x = block->Forward(past, x, mask_);
  }

  x = ln_f_->Forward(x);
  return x;
}

Tensor GPT2Model::Logits(TensorCRef hidden_state) const {
  Operators *F = ctx_.F();
  return F->MatMul(hidden_state, wte_.Transpose(0, 1));
}

}  // namespace nn
}  // namespace llama
