#include "flint/embedding.h"

#include "flint/operators.h"

namespace flint {

constexpr char Embedding::kWeight[];
std::unique_ptr<Embedding> Embedding::create(const Context &ctx, int dModel, int vocabSize) {
  std::unique_ptr<Embedding> layer{new Embedding()};
  layer->_ctx = ctx;
  layer->_dModel = dModel;
  layer->_vocabSize = vocabSize;

  return layer;
}

void Embedding::initParameters(const TensorMap &stateDict) {
  std::string nameW = _ctx.name(kWeight);

  _wte = stateDict.getTensor(nameW);
  _wte.throwIfInvalidShape({_vocabSize, _dModel});
}

Tensor Embedding::forward(const Tensor &input) const {
  Operators *F = _ctx.F();
  Tensor x = F->lookup(_wte, input);

  return x;
}

}  // namespace flint
