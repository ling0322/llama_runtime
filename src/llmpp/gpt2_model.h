#pragma once

#include <memory>
#include "flint/nn.h"
#include "llmpp/transformer.h"
#include "llyn/ini_config.h"

namespace llmpp {

// Options for GPT2 model.
struct GPT2Config {
  // config section in ini
  static constexpr char kSection[] = "gpt2";

  int nEmbd;
  int nCtx;
  int nInner;
  int nHead;
  int nLayer;
  int vocabSize;
  int hiddenSize;

  GPT2Config();
  static std::unique_ptr<GPT2Config> fromIni(const ly::IniConfig &ini);
};

// GPT2Block.
class GPT2Block : public flint::Module {
 public:
  // create BloomModel.
  static std::unique_ptr<GPT2Block> create(const flint::Context &ctx, GPT2Config config);

  // initialize the module from context
  void initParameters(const flint::TensorMap &stateDict) override;

  // forward with cache.
  // Args:
  //     cache: attention past key and value tensor cache.
  //     input <float>(N, L, C): hidden states.
  //     mask <float>(L, L): attention mask.
  // Returns:
  //     <float>(N, L, C): hidden states.
  flint::Tensor forward(
      flint::TensorMap *past, flint::TensorCRef input, flint::TensorCRef mask) const;

 private:
  flint::Context _ctx;
  GPT2Config _config;

  static constexpr char kLn1[] = "ln1";
  static constexpr char kLn2[] = "ln2";
  static constexpr char kFc[] = "fc";
  static constexpr char kProj[] = "proj";
  static constexpr char kAttn[] = "attn";

  std::unique_ptr<flint::LayerNorm> _ln1;
  std::unique_ptr<flint::LayerNorm> _ln2;
  std::unique_ptr<flint::Linear> _fc;
  std::unique_ptr<flint::Linear> _proj;
  std::unique_ptr<MultiheadSelfAttention> _attn;

  GPT2Block();
};

// The GPT2 model.
class GPT2Model : public flint::Module,
                  public flint::LanguageModel {
 public:
  // create BloomModel.
  static std::unique_ptr<GPT2Model> create(const flint::Context &ctx, GPT2Config config);

  // initialize the module from context
  void initParameters(const flint::TensorMap &stateDict) override;

  // forward with cache.
  // Args:
  //     cache: attention past key and value tensor cache.
  //     input <long>(B, L): input tokens.
  // Returns:
  //     <float>(B, L, D): logprobs.
  flint::Tensor forward(flint::TensorMap *past, flint::TensorCRef input) const override;

  // Forward the hidden state from last layer and get the logits. hidden_state
  // is usually the return value of Forward().
  // Args:
  //   hidden_state <float>(N, L, D): hidden state from last layer.
  // Returns:
  //   <float>(N, L, V): logits. V is vocabulary size.
  flint::Tensor logits(flint::TensorCRef hiddenState) const override;

 private:
  flint::Context _ctx;
  GPT2Config _config;

  static constexpr char kGpt2[] = "gpt2";
  static constexpr char kWte[] = "wte";
  static constexpr char kWpe[] = "wpe";
  static constexpr char kLnF[] = "ln_f";
  static constexpr char kBlock[] = "block";
  static constexpr char kSeqLen[] = "seq_len";

  flint::Tensor _wte;  // word embedding table
  flint::Tensor _wpe;  // positional embedding table
  flint::Tensor _mask;

  std::vector<std::unique_ptr<GPT2Block>> _blocks;
  std::unique_ptr<flint::LayerNorm> _ln;

  GPT2Model();
};

}  // namespace llmpp
