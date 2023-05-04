#ifndef GLM_RUNTIME_TRANSFORMER_DECODER_H_
#define GLM_RUNTIME_TRANSFORMER_DECODER_H_

#include <vector>
#include "nn.h"

namespace llama {
namespace nn {

class MultiheadAttention : public Module {
 public:
  // create multi-head attention module from context. 
  static StatusOr<MultiheadAttention> Create(
      const Context &ctx,
      int num_heads,
      int d_model);

  // initialize the module from context
  Status InitParameters(const TensorDict &state_dict) override;

  Tensor Forward(const Tensor &q,
                 const Tensor &k,
                 const Tensor &v,
                 const Tensor &mask = Tensor());

 private:
  int d_model_;
  int d_k_;
  int num_heads_;

  std::unique_ptr<Linear> q_proj_;
  std::unique_ptr<Linear> k_proj_;
  std::unique_ptr<Linear> v_proj_;
  std::unique_ptr<Linear> out_proj_;

  MultiheadAttention();

  Tensor Attention(const Tensor &q,
                   const Tensor &k,
                   const Tensor &v,
                   const Tensor &mask);
};


}  // namespace nn
}  // namespace llama

#endif  // GLM_RUNTIME_TRANSFORMER_DECODER_H_

