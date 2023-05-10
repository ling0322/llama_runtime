#include "test_helper.h"
#include "transformer.h"

#include "nn_test_helper.h"
#include "operators.h"
#include "util.h"

using namespace llama;
using namespace nn;


TEST_CASE("test MultiHeadAttention module", "[core][nn][transformer]") {
  util::Path model_dir = util::Path("data") / "test";
  Context ctx = MustGetCtxForCPU();

  // linear
  util::Path model_path = model_dir / "attn-model.params.bin";
  util::Path tensor_file = model_dir / "attn-model.test_tensors.bin";

  auto attn = MultiheadAttention::Create(ctx, kNumHeads, kDModel0);
  REQUIRE(attn.ok());

  MustReadParameters(model_path.string(), attn.get());
  std::vector<Tensor> tensors = MustReadAllTensors(tensor_file.string());
  REQUIRE(tensors.size() % 2 == 0);

  Tensor q = tensors[0];
  Tensor k = tensors[1];
  Tensor v = tensors[2];
  Tensor o_ref = tensors[3];

  Tensor o = attn->Forward(q, k, v);
  REQUIRE(ctx.F()->AllClose(o, o_ref));
}
