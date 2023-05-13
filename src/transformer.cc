#include "transformer.h"
#include "nn.h"
#include "operators.h"

namespace llama {
namespace nn {

MultiheadSelfAttention::MultiheadSelfAttention()
    : d_model_(0),
      d_k_(0),
      num_heads_(0) {}

StatusOr<MultiheadSelfAttention> MultiheadSelfAttention::Create(
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

  auto q_proj = Linear::Create(ctx.WithName(kQProj), d_model, d_model);
  auto k_proj = Linear::Create(ctx.WithName(kKProj), d_model, d_model);
  auto v_proj = Linear::Create(ctx.WithName(kVProj), d_model, d_model);
  auto out_proj = Linear::Create(ctx.WithName(kOutProj), d_model, d_model);

  RETURN_IF_ERROR(q_proj);
  RETURN_IF_ERROR(k_proj);
  RETURN_IF_ERROR(v_proj);
  RETURN_IF_ERROR(out_proj);

  layer->q_proj_ = std::move(q_proj).unique_ptr();
  layer->k_proj_ = std::move(k_proj).unique_ptr();
  layer->v_proj_ = std::move(v_proj).unique_ptr();
  layer->out_proj_ = std::move(out_proj).unique_ptr();

  return layer;
}

Status MultiheadSelfAttention::InitParameters(const TensorMap &state_dict) {
  RETURN_IF_ERROR(q_proj_->InitParameters(state_dict));
  RETURN_IF_ERROR(k_proj_->InitParameters(state_dict));
  RETURN_IF_ERROR(v_proj_->InitParameters(state_dict));
  RETURN_IF_ERROR(out_proj_->InitParameters(state_dict));

  return OkStatus();
}

Tensor MultiheadSelfAttention::Attention(const Tensor &q, const Tensor &k,
                                     const Tensor &v, const Tensor &mask) {
  Operators *F = ctx_.F();

  Tensor scores = F->MatMul(q, k.Transpose(-2, -1));
  scores = F->Mul(scores,  1.0f / sqrtf(1.0f * d_k_));

  if (!mask.empty()) {
      F->Print(mask);
    scores = F->Add(scores, mask);
  }

  scores = F->Softmax(scores);
  Tensor outputs = F->MatMul(scores, v);
  return outputs;
}

Tensor MultiheadSelfAttention::Forward(TensorMap *past,
                                       CTensorRef inputs,
                                       CTensorRef attn_mask) {
  Operators *F = ctx_.F();

  CHECK(inputs.dim() == 3);
  CHECK(attn_mask.empty() || attn_mask.dim() == 2);

  int bs = inputs.shape(0);
  Tensor q_proj = q_proj_->Forward(inputs);
  Tensor k_proj = k_proj_->Forward(inputs);
  Tensor v_proj = v_proj_->Forward(inputs);

  // update k_proj and v_proj according to the kv_cache from past.
  int past_len = 0;
  if (past && past->exists(pastk_name_) && past->exists(pastv_name_)) {
    CTensorRef past_k = past->Get(pastk_name_);
    CTensorRef past_v = past->Get(pastv_name_);
    past_len = past_k.shape(1);

    k_proj = F->Cat(past_k, k_proj, 1);
    v_proj = F->Cat(past_v, v_proj, 1);
    
    CHECK(k_proj.shape(1) == v_proj.shape(1));
  }

  // update kv_cache in past.
  if (past) {
    past->Put(pastk_name_, k_proj);
    past->Put(pastv_name_, v_proj);
  }

  q_proj = q_proj.View({bs, -1, num_heads_, d_k_}).Transpose(1, 2);
  k_proj = k_proj.View({bs, -1, num_heads_, d_k_}).Transpose(1, 2);
  v_proj = v_proj.View({bs, -1, num_heads_, d_k_}).Transpose(1, 2);

  Tensor mask = attn_mask.Subtensor(past_len, past_len + q_proj.shape(1));
  Tensor scores = Attention(q_proj, k_proj, v_proj, mask);
  scores = scores.Transpose(1, 2);

  Tensor concat = F->Contiguous(scores).View({bs, -1, d_model_});
  Tensor output = out_proj_->Forward(concat);
  return output;
}

}  // namespace nn
}  // namespace llama
