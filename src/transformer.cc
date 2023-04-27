#include "transformer.h"
#include "nn.h"

namespace llama {
namespace nn {

Tensor MultiheadAttention::Attention(const Tensor &q, const Tensor &k,
                                     const Tensor &v, const Tensor *mask) {
  Tensor scores = F.MatMul(q, k.Transpose(-2, -1));
  scores = F.Mul(scores,  1.0 / d_k_);
  if (mask) {
    scores = F.Add(scores, *mask);
  }

  scores = F.Softmax(scores);
  Tensor outputs = F.MatMul(scores, v);
  return outputs;
}

Tensor MultiheadAttention::Forward(const Tensor &q, const Tensor &k,
                                   const Tensor &v, const Tensor *mask) {
  CHECK(q.rank() == 3);
  CHECK(k.rank() == 3);
  CHECK(v.rank() == 3);

  int bs = q.shape(0);
  Tensor q_proj = q_proj_.Forward(q);
  Tensor k_proj = k_proj_.Forward(k);
  Tensor v_proj = v_proj_.Forward(v);

  q_proj = q_proj.View({bs, -1, num_heads_, d_k_}).Transpose(1, 2);
  k_proj = k_proj.View({bs, -1, num_heads_, d_k_}).Transpose(1, 2);
  v_proj = v_proj.View({bs, -1, num_heads_, d_k_}).Transpose(1, 2);

  Tensor scores = Attention(q_proj, k_proj, v_proj, mask);
  scores.Transpose(1, 2);

  Tensor concat = F.Contiguous(scores).View({bs, -1, d_model_});
  Tensor output = out_proj_.Forward(concat);
  return output;
}

}  // namespace nn
}  // namespace llama
