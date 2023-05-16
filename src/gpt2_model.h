#ifndef LLM_RUNTIME_BLOOM_MODEL_H_
#define LLM_RUNTIME_BLOOM_MODEL_H_

#include <memory>
#include "nn.h"

namespace llama {

class IniParser;

namespace nn {

// Options for BLOOM model.
struct GPT2Config {
  int d_model;
  int vocab_size;

  GPT2Config();
  static StatusOr<GPT2Config> FromIni(const IniParser &ini);
};

// The BLOOM model from BigScience.
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

  static constexpr char kWte[] = "wte";

  Tensor wte_;  // embedding table

  GPT2Model();
};

}  // namespace nn
}  // namespace llama

#endif  // LLM_RUNTIME_BLOOM_MODEL_H_
