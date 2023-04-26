#include "transformer.h"
#include "nn.h"

namespace llama {
namespace nn {

Tensor MultiheadAttention::Attention(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    const Tensor *mask) {
  Tensor k_T = F.Transpose(k, -2, -1);
  Tensor scores = F.BMM(q, k_T);
  scores = F.Mul(scores,  1.0 / d_k_);
  if (mask) {
    scores = F.Add(scores, *mask);
  }

  scores = F.Softmax(scores);
  Tensor outputs = F.BMM(scores, v);
  return outputs;
}

Tensor MultiheadAttention::Forward(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    const Tensor *mask_in = nullptr) {
  CHECK(q.rank() == 3);
  CHECK(k.rank() == 3);
  CHECK(v.rank() == 3);

  int batch_size = q.shape(0);


}

}  // namespace nn
}  // namespace llama


