#include "transformer.h"
#include "nn.h"
#include "operators.h"

namespace llama {
namespace nn {

StatusOr<MultiheadAttention> MultiheadAttention::Create(
    const Context &ctx,
    int num_heads,
    int d_model) {
  std::unique_ptr<MultiheadAttention> layer{new MultiheadAttention()};
  if (d_model % num_heads != 0) {
    RETURN_ABORTED() << "invalid d_model and num_heads";
  }

  layer->d_model_ = d_model;
  layer->d_k_ = d_model / num_heads;
  layer->num_heads_ = num_heads;

  auto q_proj = Linear::Create(ctx.WithName("q_proj"), d_model, d_model);
  auto k_proj = Linear::Create(ctx.WithName("k_proj"), d_model, d_model);
  auto v_proj = Linear::Create(ctx.WithName("v_proj"), d_model, d_model);
  auto out_proj = Linear::Create(ctx.WithName("out_proj"), d_model, d_model);

  RETURN_IF_ERROR(q_proj);
  RETURN_IF_ERROR(k_proj);
  RETURN_IF_ERROR(v_proj);
  RETURN_IF_ERROR(out_proj);

  layer->q_proj_ = std::move(q_proj).pointer();
  layer->k_proj_ = std::move(k_proj).pointer();
  layer->v_proj_ = std::move(v_proj).pointer();
  layer->out_proj_ = std::move(out_proj).pointer();

  return layer;
}

Tensor MultiheadAttention::Attention(const Tensor &q, const Tensor &k,
                                     const Tensor &v, const Tensor &mask) {
  Operators *F = ctx_.F();

  Tensor scores = F->MatMul(q, k.Transpose(-2, -1));
  scores = F->Mul(scores,  1.0 / d_k_);
  if (!mask.empty()) {
    scores = F->Add(scores, mask);
  }

  scores = F->Softmax(scores);
  Tensor outputs = F->MatMul(scores, v);
  return outputs;
}

Tensor MultiheadAttention::Forward(const Tensor &q, const Tensor &k,
                                   const Tensor &v, const Tensor &mask) {
  Operators *F = ctx_.F();

  CHECK(q.rank() == 3);
  CHECK(k.rank() == 3);
  CHECK(v.rank() == 3);
  CHECK(mask.empty() || mask.rank() == 2);

  int bs = q.shape(0);
  Tensor q_proj = q_proj_->Forward(q);
  Tensor k_proj = k_proj_->Forward(k);
  Tensor v_proj = v_proj_->Forward(v);

  q_proj = q_proj.View({bs, -1, num_heads_, d_k_}).Transpose(1, 2);
  k_proj = k_proj.View({bs, -1, num_heads_, d_k_}).Transpose(1, 2);
  v_proj = v_proj.View({bs, -1, num_heads_, d_k_}).Transpose(1, 2);

  Tensor scores = Attention(q_proj, k_proj, v_proj, mask);
  scores.Transpose(1, 2);

  Tensor concat = F->Contiguous(scores).View({bs, -1, d_model_});
  Tensor output = out_proj_->Forward(concat);
  return output;
}

}  // namespace nn
}  // namespace llama
