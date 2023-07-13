#pragma once

#include <vector>
#include "flint/nn.h"
#include "flint/tensor.h"

namespace llmpp {

class MultiheadSelfAttention : public flint::Module {
 public:
  // create multi-head attention module from context. 
  static std::unique_ptr<MultiheadSelfAttention> create(
      const flint::Context &ctx, int numHeads, int dModel);

  // initialize the module from context
  void initParameters(const flint::TensorMap &stateDict) override;

  // forward the inputs into multi-head self-attention. inputs was both q, k and v for the attention
  // module. If past is not nullptr, it will concat kv_cache from past before compute attention,
  // then put the updated kv_cache back to past.
  flint::Tensor forward(
      flint::TensorMap *past, const flint::Tensor &inputs,  const flint::Tensor &mask);

 private:
  flint::Context _ctx;

  int _dModel;
  int _dK;
  int _numHeads;

  std::string _namePastK;
  std::string _namePastV;

  static constexpr char kQProj[] = "q_proj";
  static constexpr char kKProj[] = "k_proj";
  static constexpr char kVProj[] = "v_proj";
  static constexpr char kOutProj[] = "out_proj";

  std::unique_ptr<flint::Linear> _qProj;
  std::unique_ptr<flint::Linear> _kProj;
  std::unique_ptr<flint::Linear> _vProj;
  std::unique_ptr<flint::Linear> _outProj;

  MultiheadSelfAttention();

  flint::Tensor attention(
      const flint::Tensor &q,
      const flint::Tensor &k,
      const flint::Tensor &v,
      const flint::Tensor &mask);
};

}  // namespace llmpp
