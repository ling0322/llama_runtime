#ifndef LLM_RUNTIME_BLOOM_MODEL_H_
#define LLM_RUNTIME_BLOOM_MODEL_H_

#include <memory>
#include "nn.h"
#include "transformer.h"

namespace llama {

class IniParser;

namespace nn {

// Options for BLOOM model.
struct GPT2Config {
  int n_embd;
  int n_ctx;
  int n_inner;
  int n_head;
  int vocab_size;
  int hidden_size;

  GPT2Config();
  static StatusOr<GPT2Config> FromIni(const IniParser &ini);
};

// GPT2Block.
class GPT2Block : public Module {
 public:
  // create BloomModel.
  static StatusOr<GPT2Block> Create(const Context &ctx, GPT2Config config);

  // initialize the module from context
  Status InitParameters(const TensorMap &state_dict) override;

  // forward with cache.
  // Args:
  //     cache: attention past key and value tensor cache.
  //     input <float>(N, L, C): hidden states.
  //     mask <float>(L, L): attention mask.
  // Returns:
  //     <float>(N, L, C): hidden states.
  Tensor Forward(TensorMap *past, TensorCRef input, TensorCRef mask) const;

 private:
  GPT2Config config_;

  static constexpr char kLn1[] = "ln1";
  static constexpr char kLn2[] = "ln2";
  static constexpr char kFc[] = "fc";
  static constexpr char kProj[] = "proj";
  static constexpr char kAttn[] = "attn";

  std::unique_ptr<LayerNorm> ln1_;
  std::unique_ptr<LayerNorm> ln2_;
  std::unique_ptr<Linear> fc_;
  std::unique_ptr<Linear> proj_;
  std::unique_ptr<MultiheadSelfAttention> attn_;

  GPT2Block();
};

// The GPT2 model.
class GPT2Model : public Module {
 public:
  // create BloomModel.
  static StatusOr<GPT2Model> Create(const Context &ctx, GPT2Config config);

  // initialize the module from context
  Status InitParameters(const TensorMap &state_dict) override;

  // forward with cache.
  // Args:
  //     cache: attention past key and value tensor cache.
  //     input <long>(B, L): input tokens.
  // Returns:
  //     <float>(B, L, D): logprobs.
  Tensor Forward(TensorMap *past, TensorCRef input) const;

 private:
  GPT2Config config_;

  static constexpr char kGpt2[] = "gpt2";
  static constexpr char kWte[] = "wte";
  static constexpr char kWpe[] = "wpe";
  static constexpr char kBlock[] = "block0";

  Tensor wte_;  // word embedding table
  Tensor wpe_;  // positional embedding table
  Tensor mask_;
  std::unique_ptr<GPT2Block> block_;

  GPT2Model();
};

}  // namespace nn
}  // namespace llama

#endif  // LLM_RUNTIME_BLOOM_MODEL_H_
