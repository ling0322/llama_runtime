#include "bloom_model.h"

#include <memory>

namespace llama {
namespace nn {

// ---------------------------------------------------------------------------+
// class BloomModelOptions                                                    |
// ---------------------------------------------------------------------------+

BloomModelOptions::BloomModelOptions()
    : d_model(0),
      vocab_size(0) {
}

// ---------------------------------------------------------------------------+
// class BloomModel                                                           |
// ---------------------------------------------------------------------------+

StatusOr<BloomModel> BloomModel::Create(const Context &ctx,
                                        BloomModelOptions options) {
  std::unique_ptr<BloomModel> model{new BloomModel()};

  model->options_ = options;
  if (options.d_model <= 0) {
    RETURN_ABORTED() << "invalid d_model";
  }

  StatusOr<LayerNorm> ln_emb = LayerNorm::Create(ctx, options.d_model);
  RETURN_IF_ERROR(ln_emb);

  model->ln_emb_ = std::move(ln_emb).pointer();
  return model;
}

Status InitParameters(const TensorDict &state_dict) override;

}  // namespace nn
}  // namespace llama
