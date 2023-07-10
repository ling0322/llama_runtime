#include "catch2/catch_amalgamated.hpp"
#include "llmrt/transformer.h"
#include "flint/operators.h"
#include "flint/util.h"
#include "llyn/path.h"

using namespace flint;
using namespace llmrt;
using namespace ly;

constexpr int kDModel0 = 16;
constexpr int kDModel1 = 20;
constexpr int kSeqLen = 10;
constexpr int kBatchSize = 2;
constexpr int kNumHeads = 2;

TEST_CASE("test self-attention module", "[core][nn][transformer]") {
  ly::Path model_dir = ly::Path("data") / "test";
  Context ctx = getCtxForCPU();

  // linear
  ly::Path model_path = model_dir / "self-attn.params.bin";
  ly::Path tensor_file = model_dir / "self-attn.test_tensors.bin";

  auto attn = MultiheadSelfAttention::create(ctx, kNumHeads, kDModel0);
  readParameters(model_path.string(), attn.get());
  std::vector<Tensor> tensors = readAllTensors(tensor_file.string());
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
