#ifndef GLM_RUNTIME_TRANSFORMER_DECODER_H_
#define GLM_RUNTIME_TRANSFORMER_DECODER_H_

#include <vector>
#include "nn.h"

namespace llama {
namespace nn {

class MultiheadAttention : public Module {
 public:
  MultiheadAttention(Namespace ns, int d_model, int num_heads);

  Tensor Forward(const Tensor &q,
                 const Tensor &k,
                 const Tensor &v,
                 const Tensor *mask = nullptr);

 private:
  Namespace ns;
  Function F;

  int d_model_;
  int d_k_;
  int num_heads_;

  Linear q_proj_;
  Linear k_proj_;
  Linear v_proj_;
  Linear out_proj_;

  Tensor Attention(const Tensor &q,
                   const Tensor &k,
                   const Tensor &v,
                   const Tensor *mask = nullptr);
};


}  // namespace nn
}  // namespace llama

#endif  // GLM_RUNTIME_TRANSFORMER_DECODER_H_

