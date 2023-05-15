#include "test_helper.h"
#include "transformer.h"

#include "nn_test_helper.h"
#include "operators.h"
#include "util.h"

using namespace llama;
using namespace nn;


TEST_CASE("test self-attention module", "[core][nn][transformer]") {
  util::Path model_dir = util::Path("data") / "test";
  Context ctx = MustGetCtxForCPU();

  // linear
  util::Path model_path = model_dir / "self-attn.params.bin";
  util::Path tensor_file = model_dir / "self-attn.test_tensors.bin";

  auto attn = MultiheadSelfAttention::Create(ctx, kNumHeads, kDModel0);
  REQUIRE(attn.ok());

  MustReadParameters(model_path.string(), attn.get());
  std::vector<Tensor> tensors = MustReadAllTensors(tensor_file.string());
  REQUIRE(tensors.size() % 2 == 0);

  Tensor inputs = tensors[0];
  Tensor o_ref = tensors[1];

  Tensor mask = ctx.F()->CausalMask(kSeqLen);

  Tensor o = attn->Forward(nullptr, inputs, mask);
  ctx.F()->Print(o);
  ctx.F()->Print(o_ref);
  REQUIRE(ctx.F()->AllClose(o, o_ref));

  puts("------------------------");

  // autoregressive mode
  TensorMap kv_cache;
  Tensor x = inputs.Slice(1, 0, 5);  // tensor x is in NWC format
  ctx.F()->Print(x);
  o = attn->Forward(&kv_cache, inputs, mask);
  ctx.F()->Print(o);
  for (int i = 5; i < kSeqLen; ++i) {
    x = inputs.Slice(1, i, i + 1);
    o = ctx.F()->Cat(o, attn->Forward(&kv_cache, x, mask), 1);
    printf("%d\n", i);
    ctx.F()->Print(o);
  }

  ctx.F()->Print(o);
  ctx.F()->Print(o_ref);

  REQUIRE(ctx.F()->AllClose(o, o_ref));
}
