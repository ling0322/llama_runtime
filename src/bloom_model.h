#ifndef LLM_RUNTIME_BLOOM_MODEL_H_
#define LLM_RUNTIME_BLOOM_MODEL_H_

#include <memory>
#include "nn.h"

namespace llama {
namespace nn {

// Options for BLOOM model.
struct BloomModelOptions {
  int d_model;
  int vocab_size;

  BloomModelOptions();
};

// The BLOOM model from BigScience.
class BloomModel : public Module {
 public:
  // create BloomModel.
  static StatusOr<BloomModel> Create(const Context &ctx,
                                     BloomModelOptions options);

  // initialize the module from context
  Status InitParameters(const TensorDict &state_dict) override;

  // forward with cache.
  // Args:
  //     cache: attention past key and value tensor cache.
  //     input <long>(B, L): input tokens.
  // Returns:
  //     <float>(B, L, D): logprobs.
  Tensor Forward(TensorDict *cache, const Tensor &input) const override;

 private:
  BloomModelOptions options_;

  Tensor embedding_;
  std::unique_ptr<LayerNorm> ln_emb_;

  BloomModel();
};

}  // namespace nn
}  // namespace llama

#endif  // LLM_RUNTIME_BLOOM_MODEL_H_
