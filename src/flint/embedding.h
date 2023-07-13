#pragma once

#include "flint/nn.h"

namespace flint {

class Embedding : public Module {
 public:
  static std::unique_ptr<Embedding> create(const Context &ctx, int dModel, int vocabSize);

  // initialize the module from state dict.
  void initParameters(const TensorMap &state_dict) override;

  // forward input and return the output.
  // Args:
  //   input <long>(N, L): input word-ids.
  // Returns:
  //   <float>(N, L, D): embeddings for input word-ids.
  Tensor forward(const Tensor &input) const;

 private:
  // tensor names.
  static constexpr char kWeight[] = "weight";

  Context _ctx;
  Tensor _wte;

  int _dModel;
  int _vocabSize;

  Embedding() = default;
};

}  // namespace flint
