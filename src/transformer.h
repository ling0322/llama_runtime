#ifndef GLM_RUNTIME_TRANSFORMER_DECODER_H_
#define GLM_RUNTIME_TRANSFORMER_DECODER_H_

#include <vector>
#include "nn.h"

namespace llama {
namespace nn {

class MultiheadSelfAttention : public Module {
 public:
  // create multi-head attention module from context. 
  static std::unique_ptr<MultiheadSelfAttention> create(
      const Context &ctx, int numHeads, int dModel);

  // initialize the module from context
  void initParameters(const TensorMap &stateDict) override;

  // forward the inputs into multi-head self-attention. inputs was both q, k and v for the attention
  // module. If past is not nullptr, it will concat kv_cache from past before compute attention,
  // then put the updated kv_cache back to past.
  Tensor forward(TensorMap *past, TensorCRef inputs, TensorCRef mask);

 private:
  Context _ctx;

  int _dModel;
  int _dK;
  int _numHeads;

  std::string _namePastK;
  std::string _namePastV;

  static constexpr char kQProj[] = "q_proj";
  static constexpr char kKProj[] = "k_proj";
  static constexpr char kVProj[] = "v_proj";
  static constexpr char kOutProj[] = "out_proj";

  std::unique_ptr<Linear> _qProj;
  std::unique_ptr<Linear> _kProj;
  std::unique_ptr<Linear> _vProj;
  std::unique_ptr<Linear> _outProj;

  MultiheadSelfAttention();

  Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &mask);
};


}  // namespace nn
}  // namespace llama

#endif  // GLM_RUNTIME_TRANSFORMER_DECODER_H_

