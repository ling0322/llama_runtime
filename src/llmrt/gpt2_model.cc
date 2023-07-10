#include "llmrt/gpt2_model.h"

#include <memory>
#include "flint/operators.h"
#include "llyn/ini_config.h"
#include "llyn/strings.h"

namespace llmrt {

using flint::Context;
using flint::LayerNorm;
using flint::Linear;
using flint::TensorMap;
using flint::TensorMap;
using flint::TensorCRef;
using flint::Tensor;
using flint::Operators;
using flint::DType;
using flint::LongType;

// -- class GPT2Config ----------

GPT2Config::GPT2Config()
    : nEmbd(0),
      nCtx(0),
      nInner(0),
      nHead(0),
      nLayer(0),
      vocabSize(0),
      hiddenSize(0) {
}

std::unique_ptr<GPT2Config> GPT2Config::fromIni(const ly::IniConfig &ini) {
  auto config = std::make_unique<GPT2Config>();

  const ly::IniSection &section = ini.getSection(kSection);

  config->nEmbd = section.getInt("n_embd");
  config->nCtx = section.getInt("n_ctx");
  config->nInner = section.getInt("n_inner");
  config->nHead = section.getInt("n_head");
  config->nLayer = section.getInt("n_layer");
  config->vocabSize = section.getInt("vocab_size");
  config->hiddenSize = section.getInt("hidden_size");

  return config;
}

// -- class GPT2Block ----------

GPT2Block::GPT2Block() {}

std::unique_ptr<GPT2Block> GPT2Block::create(const Context &ctx, GPT2Config config) {
  std::unique_ptr<GPT2Block> block{new GPT2Block()};
  
  block->_ctx = ctx;
  block->_config = config;

  int dModel = config.hiddenSize;
  int nInner = config.nInner;

  auto ln1 = LayerNorm::create(ctx.withName(kLn1), dModel);
  auto ln2 = LayerNorm::create(ctx.withName(kLn2), dModel);
  auto fc = Linear::create(ctx.withName(kFc), dModel, nInner);
  auto proj = Linear::create(ctx.withName(kProj), nInner, dModel);
  auto attn = MultiheadSelfAttention::create(ctx.withName(kAttn), config.nHead, dModel);

  block->_ln2 = std::move(ln2);
  block->_ln1 = std::move(ln1);
  block->_attn = std::move(attn);
  block->_fc = std::move(fc);
  block->_proj = std::move(proj);

  return block;
}

void GPT2Block::initParameters(const TensorMap &state_dict) {
  _ln2->initParameters(state_dict);
  _ln1->initParameters(state_dict);
  _attn->initParameters(state_dict);
  _fc->initParameters(state_dict);
  _proj->initParameters(state_dict);
}

Tensor GPT2Block::forward(TensorMap *past, TensorCRef input, TensorCRef mask) const {
  Operators *F = _ctx.F();

  Tensor residual = input;
  Tensor x = _ln1->forward(input);
  x = _attn->forward(past, x, mask);
  x = F->add(x, residual);

  residual = x;
  x = _ln2->forward(x);
  x = _fc->forward(x);
  x = F->gelu(x);
  x = _proj->forward(x);
  x = F->add(x, residual);

  return x;
}

// -- class GPT2Model ----------

GPT2Model::GPT2Model() {}

std::unique_ptr<GPT2Model> GPT2Model::create(const Context &ctx, GPT2Config config) {
  std::unique_ptr<GPT2Model> model{new GPT2Model()};

  model->_ctx = ctx.withName(kGpt2);
  model->_config = config;

  for (int i = 0; i < config.nLayer; ++i) {
    std::string block_name = ly::sprintf("%s%d", kBlock, i);
    auto block = GPT2Block::create(model->_ctx.withName(block_name), config);

    model->_blocks.emplace_back(std::move(block));
  }

  auto ln = LayerNorm::create(model->_ctx.withName(kLnF), config.hiddenSize);
  model->_ln = std::move(ln);

  return model;
}

void GPT2Model::initParameters(const TensorMap &state_dict) {
  Operators *F = _ctx.F();

  _wte = state_dict.getTensor(_ctx.name(kWte));
  _wpe = state_dict.getTensor(_ctx.name(kWpe));

  _wte.throwIfInvalidShape({_config.vocabSize, _config.nEmbd});
  _wpe.throwIfInvalidShape({_config.nCtx, _config.nEmbd});

  for (const std::unique_ptr<GPT2Block> &block : _blocks) {
    block->initParameters(state_dict);
  }
  _ln->initParameters(state_dict);

  _mask = F->causalMask(_config.nCtx);
}

Tensor GPT2Model::forward(TensorMap *past, TensorCRef inputIds) const {
  CHECK(inputIds.getDim() == 2 && inputIds.getDType() == DType::kLong);

  Operators *F = _ctx.F();
  Tensor x = F->lookup(_wte, inputIds);

  // seq_len is the number of token ids processed so far. Which is used to get the correct index of
  // positional embedding for current `inputIds`. Since `past` could only accept tensors, we use a
  // scalar Tensor to store it.
  int inputStartIdx = 0;
  if (past) {
    std::string name = _ctx.name(kSeqLen);
    if (past->hasTensor(name)) {
      Tensor x = past->getTensor(name);
      inputStartIdx = static_cast<int>(x.getElem<LongType>({0}));
    }

    int next_idx = inputStartIdx + inputIds.getShape(1);
    past->putTensor(name, Tensor::create<LongType>({1}, {next_idx}));
  }
  Tensor posEmb = _wpe.slice(inputStartIdx, inputStartIdx + x.getShape(1));
  x = F->add(x, posEmb);

  int i = 0;
  for (const std::unique_ptr<GPT2Block> &block : _blocks) {
    x = block->forward(past, x, _mask);
  }

  x = _ln->forward(x);
  return x;
}

Tensor GPT2Model::logits(TensorCRef hiddenState) const {
  Operators *F = _ctx.F();
  return F->matmul(hiddenState, _wte.transpose(0, 1));
}

}  // namespace llmrt
