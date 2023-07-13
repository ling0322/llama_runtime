#include "llmpp/chatglm2_model.h"

using flint::Context;
using flint::Tensor;
using flint::TensorMap;


namespace llmpp {

constexpr char ChatGLM2Config::kSection[];

ChatGLM2Config::ChatGLM2Config()
    : nEmbd(0),
      vocabSize(0) {}

std::unique_ptr<ChatGLM2Config> ChatGLM2Config::fromIni(const ly::IniConfig &ini) {
  std::unique_ptr<ChatGLM2Config> config{new ChatGLM2Config()};

  const ly::IniSection &section = ini.getSection(kSection);

  config->nEmbd = section.getInt("n_embd");
  config->nEmbd = section.getInt("vocab_size");
  return config;
}

constexpr char ChatGLM2Model::kChatGlm2[];
constexpr char ChatGLM2Model::kEmbd[];

ChatGLM2Model::ChatGLM2Model() {}

std::unique_ptr<ChatGLM2Model> ChatGLM2Model::create(const Context &ctx, ChatGLM2Config config) {
  std::unique_ptr<ChatGLM2Model> layer{new ChatGLM2Model()};
  layer->_config = config;
  layer->_ctx = ctx;
  layer->_embedding = flint::Embedding::create(ctx.withName(kEmbd), config.nEmbd, config.vocabSize);

  return layer;
}

void ChatGLM2Model::initParameters(const TensorMap &stateDict) {
  _embedding->initParameters(stateDict);
}

Tensor ChatGLM2Model::forward(TensorMap *past, const Tensor &input) const {
  return _embedding->forward(input);
}

};
