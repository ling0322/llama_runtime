#include "transformer.h"
#include "nn.h"
#include "operators.h"

namespace llama {
namespace nn {

MultiheadSelfAttention::MultiheadSelfAttention()
    : _dModel(0),
      _dK(0),
      _numHeads(0) {}

std::unique_ptr<MultiheadSelfAttention> MultiheadSelfAttention::create(
    const Context &ctx, int numHeads, int dModel) {
  std::unique_ptr<MultiheadSelfAttention> layer{new MultiheadSelfAttention()};
  if (dModel % numHeads != 0) {
    throw AbortedException("invalid d_model and num_heads");
  }

  layer->_dModel = dModel;
  layer->_dK = dModel / numHeads;
  layer->_numHeads = numHeads;
  layer->_ctx = ctx;

  layer->_namePastK = ctx.name("k");
  layer->_namePastV = ctx.name("v");

  layer->_qProj = Linear::create(ctx.withName(kQProj), dModel, dModel);
  layer->_kProj = Linear::create(ctx.withName(kKProj), dModel, dModel);
  layer->_vProj = Linear::create(ctx.withName(kVProj), dModel, dModel);
  layer->_outProj = Linear::create(ctx.withName(kOutProj), dModel, dModel);

  return layer;
}

void MultiheadSelfAttention::initParameters(const TensorMap &state_dict) {
  _qProj->initParameters(state_dict);
  _kProj->initParameters(state_dict);
  _vProj->initParameters(state_dict);
  _outProj->initParameters(state_dict);
}

Tensor MultiheadSelfAttention::attention(
    TensorCRef q, TensorCRef k, TensorCRef v, TensorCRef mask) {
  Operators *F = _ctx.F();

  Tensor scores = F->bmm(q, k.transpose(-2, -1));
  scores = F->mul(scores,  1.0f / sqrtf(1.0f * _dK));

  if (!mask.empty()) {
    scores = F->add(scores, mask);
  }

  scores = F->softmax(scores);
  Tensor outputs = F->bmm(scores, v);
  return outputs;
}

Tensor MultiheadSelfAttention::forward(TensorMap *past, TensorCRef inputs, TensorCRef attn_mask) {
  Operators *F = _ctx.F();

  CHECK(inputs.getDim() == 3);
  CHECK(attn_mask.empty() || attn_mask.getDim() == 2);

  int bs = inputs.getShape(0);
  Tensor qProj = _qProj->forward(inputs);
  Tensor kProj = _kProj->forward(inputs);
  Tensor vProj = _vProj->forward(inputs);

  // update k_proj and v_proj according to the kv_cache from past.
  int past_len = 0;
  if (past && past->hasTensor(_namePastK) && past->hasTensor(_namePastV)) {
    TensorCRef past_k = past->getTensor(_namePastK);
    TensorCRef past_v = past->getTensor(_namePastV);
    past_len = past_k.getShape(1);

    kProj = F->cat(past_k, kProj, 1);
    vProj = F->cat(past_v, vProj, 1);
    
    CHECK(kProj.getShape(1) == vProj.getShape(1));
  }

  // update kv_cache in past.
  if (past) {
    past->putTensor(_namePastK, kProj);
    past->putTensor(_namePastV, vProj);
  }

  qProj = qProj.view({bs, -1, _numHeads, _dK}).transpose(1, 2);
  kProj = kProj.view({bs, -1, _numHeads, _dK}).transpose(1, 2);
  vProj = vProj.view({bs, -1, _numHeads, _dK}).transpose(1, 2);

  Tensor mask = attn_mask.empty() 
      ? attn_mask 
      : attn_mask.slice(0, past_len, past_len + qProj.getShape(2)).slice(1, 0, kProj.getShape(2));
  Tensor scores = attention(qProj, kProj, vProj, mask);
  scores = scores.transpose(1, 2);

  Tensor concat = F->contiguous(scores).view({bs, -1, _dModel});
  Tensor output = _outProj->forward(concat);
  return output;
}

}  // namespace nn
}  // namespace llama
