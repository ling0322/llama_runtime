#ifndef GLM_RUNTIME_TRANSFORMER_DECODER_H_
#define GLM_RUNTIME_TRANSFORMER_DECODER_H_

#include <vector>
#include "nn.h"

namespace llama {
namespace nn {

class MultiheadAttention {
 public:
 private:
  Tensor Attention(const Tensor &q,
                   const Tensor &k,
                   const Tensor &v,
                   const Tensor *mask);
};

Tensor Function::Attention(const Tensor &q, const Tensor &k, const Tensor &v,
                           const Tensor *mask) {
  float d_k = q.shape(-1);
  Tensor k_T = Transpose(k, 2, 3);

  Tensor scores = MatMul(q, k_T);
  scores = Scale(scores, d_k);
  if (mask) {
    scores = Add(scores, *mask);
  }

  scores = Softmax(scores);
  Tensor outputs = MatMul(scores, v);
}


}  // namespace nn
}  // namespace llama

#endif  // GLM_RUNTIME_TRANSFORMER_DECODER_H_

