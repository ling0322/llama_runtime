#include "transformer.h"
#include "nn.h"
#include "operators.h"

namespace llama {
namespace nn {

MultiheadSelfAttention::MultiheadSelfAttention()
    : d_model_(0),
      d_k_(0),
      num_heads_(0) {}

expected_ptr<MultiheadSelfAttention> MultiheadSelfAttention::Create(
    const Context &ctx,
    int num_heads,
    int d_model) {
  std::unique_ptr<MultiheadSelfAttention> layer{new MultiheadSelfAttention()};
  if (d_model % num_heads != 0) {
    RETURN_ABORTED() << "invalid d_model and num_heads";
  }

  layer->d_model_ = d_model;
  layer->d_k_ = d_model / num_heads;
  layer->num_heads_ = num_heads;
  layer->ctx_ = ctx;

  layer->pastk_name_ = ctx.name("k");
  layer->pastv_name_ = ctx.name("v");

  auto q_proj = Linear::create(ctx.withName(kQProj), d_model, d_model);
  auto k_proj = Linear::create(ctx.withName(kKProj), d_model, d_model);
  auto v_proj = Linear::create(ctx.withName(kVProj), d_model, d_model);
  auto out_proj = Linear::create(ctx.withName(kOutProj), d_model, d_model);

  layer->q_proj_ = std::move(q_proj);
  layer->k_proj_ = std::move(k_proj);
  layer->v_proj_ = std::move(v_proj);
  layer->out_proj_ = std::move(out_proj);

  return layer;
}

void MultiheadSelfAttention::initParameters(const TensorMap &state_dict) {
  q_proj_->initParameters(state_dict);
  k_proj_->initParameters(state_dict);
  v_proj_->initParameters(state_dict);
  out_proj_->initParameters(state_dict);
}

Tensor MultiheadSelfAttention::Attention(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    const Tensor &mask) {
  Operators *F = ctx_.F();

  Tensor scores = F->matmul(q, k.transpose(-2, -1));
  scores = F->mul(scores,  1.0f / sqrtf(1.0f * d_k_));

  if (!mask.empty()) {
    scores = F->add(scores, mask);
  }

  scores = F->softmax(scores);
  Tensor outputs = F->matmul(scores, v);
  return outputs;
}

Tensor MultiheadSelfAttention::Forward(TensorMap *past,
                                       TensorCRef inputs,
                                       TensorCRef attn_mask) {
  Operators *F = ctx_.F();

  CHECK(inputs.getDim() == 3);
  CHECK(attn_mask.empty() || attn_mask.getDim() == 2);

  int bs = inputs.getShape(0);
  Tensor q_proj = q_proj_->forward(inputs);
  Tensor k_proj = k_proj_->forward(inputs);
  Tensor v_proj = v_proj_->forward(inputs);

  // update k_proj and v_proj according to the kv_cache from past.
  int past_len = 0;
  if (past && past->hasTensor(pastk_name_) && past->hasTensor(pastv_name_)) {
    TensorCRef past_k = past->getTensor(pastk_name_);
    TensorCRef past_v = past->getTensor(pastv_name_);
    past_len = past_k.getShape(1);

    k_proj = F->cat(past_k, k_proj, 1);
    v_proj = F->cat(past_v, v_proj, 1);
    
    CHECK(k_proj.getShape(1) == v_proj.getShape(1));
  }

  // update kv_cache in past.
  if (past) {
    past->putTensor(pastk_name_, k_proj);
    past->putTensor(pastv_name_, v_proj);
  }

  q_proj = q_proj.view({bs, -1, num_heads_, d_k_}).transpose(1, 2);
  k_proj = k_proj.view({bs, -1, num_heads_, d_k_}).transpose(1, 2);
  v_proj = v_proj.view({bs, -1, num_heads_, d_k_}).transpose(1, 2);

  Tensor mask = attn_mask.empty() 
      ? attn_mask 
      : attn_mask.slice(0, past_len, past_len + q_proj.getShape(2))
                 .slice(1, 0, k_proj.getShape(2));
  Tensor scores = Attention(q_proj, k_proj, v_proj, mask);
  scores = scores.transpose(1, 2);

  Tensor concat = F->contiguous(scores).view({bs, -1, d_model_});
  Tensor output = out_proj_->forward(concat);
  return output;
}

}  // namespace nn
}  // namespace llama
