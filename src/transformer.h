#ifndef GLM_RUNTIME_TRANSFORMER_DECODER_H_
#define GLM_RUNTIME_TRANSFORMER_DECODER_H_

#include <vector>
#include "nn.h"

namespace llama {
namespace nn {

class MultiheadSelfAttention : public Module {
 public:
  // create multi-head attention module from context. 
  static expected_ptr<MultiheadSelfAttention> Create(
      const Context &ctx,
      int num_heads,
      int d_model);

  // initialize the module from context
  Status InitParameters(const TensorMap &state_dict) override;

  // forward the inputs into multi-head self-attention. inputs was both q, k
  // and v for the attention module.
  // If past is not nullptr, it will concat kv_cache from past before compute
  // attention, then put the updated kv_cache back to past.
  Tensor Forward(TensorMap *past, TensorCRef inputs, TensorCRef mask);

 private:
  Context ctx_;

  int d_model_;
  int d_k_;
  int num_heads_;

  std::string pastk_name_;
  std::string pastv_name_;

  static constexpr char kQProj[] = "q_proj";
  static constexpr char kKProj[] = "k_proj";
  static constexpr char kVProj[] = "v_proj";
  static constexpr char kOutProj[] = "out_proj";

  std::unique_ptr<Linear> q_proj_;
  std::unique_ptr<Linear> k_proj_;
  std::unique_ptr<Linear> v_proj_;
  std::unique_ptr<Linear> out_proj_;

  MultiheadSelfAttention();

  Tensor Attention(const Tensor &q,
                   const Tensor &k,
                   const Tensor &v,
                   const Tensor &mask);
};


}  // namespace nn
}  // namespace llama

#endif  // GLM_RUNTIME_TRANSFORMER_DECODER_H_

