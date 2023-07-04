#include "common/test_helper.h"
#include "model/transformer.h"

#include "nn/nn_test_helper.h"
#include "nn/operators.h"
#include "util/util.h"

using namespace llama;
using namespace nn;


TEST_CASE("test self-attention module", "[core][nn][transformer]") {
  util::Path model_dir = util::Path("data") / "test";
  Context ctx = MustGetCtxForCPU();

  // linear
  util::Path model_path = model_dir / "self-attn.params.bin";
  util::Path tensor_file = model_dir / "self-attn.test_tensors.bin";

  auto attn = MultiheadSelfAttention::create(ctx, kNumHeads, kDModel0);
  MustReadParameters(model_path.string(), attn.get());
  std::vector<Tensor> tensors = MustReadAllTensors(tensor_file.string());
  REQUIRE(tensors.size() % 2 == 0);

  Tensor inputs = tensors[0];
  Tensor o_ref = tensors[1];

  Tensor mask = ctx.F()->causalMask(kSeqLen);

  Tensor o = attn->forward(nullptr, inputs, mask);
  REQUIRE(ctx.F()->allClose(o, o_ref));

  // autoregressive mode
  TensorMap kv_cache;
  Tensor x = inputs.slice(1, 0, 5);  // tensor x is in NWC format
  o = attn->forward(&kv_cache, x, mask);
  for (int i = 5; i < kSeqLen; ++i) {
    x = inputs.slice(1, i, i + 1);
    o = ctx.F()->cat(o, attn->forward(&kv_cache, x, mask), 1);
  }

  REQUIRE(ctx.F()->allClose(o, o_ref));
}
