#include "test_helper.h"
#include "transformer.h"

#include "nn_test_helper.h"
#include "operators.h"
#include "util.h"

using namespace llama;
using namespace nn;

TEST_CASE("test subtensor and slice", "[core][nn][tensor]") {
  Context ctx = MustGetCtxForCPU();
  Tensor tensor = MakeTensor(ctx.F(), {4, 4}, {
    0.0f, 0.1f, 0.2f, 0.3f,
    0.4f, 0.5f, 0.6f, 0.7f,
    0.8f, 0.9f, 1.0f, 1.1f,
    1.2f, 1.3f, 1.4f, 1.5f
  });

  // slice (dim 0)
  Tensor subtensor = MakeTensor(ctx.F(), {2, 4}, {
    0.4f, 0.5f, 0.6f, 0.7f,
    0.8f, 0.9f, 1.0f, 1.1f,
  });
  REQUIRE(ctx.F()->AllClose(tensor.Slice(1, 3), subtensor));
  
  // subtensor
  subtensor = MakeTensor(ctx.F(), {4}, {
    0.4f, 0.5f, 0.6f, 0.7f,
  });
  REQUIRE(ctx.F()->AllClose(tensor.Subtensor(1), subtensor));

  // slice (any dim)
  subtensor = MakeTensor(ctx.F(), { 2, 2 }, {
    0.5f, 0.6f,
    0.9f, 1.0f,
  });
  REQUIRE(ctx.F()->AllClose(
      tensor.Slice(0, 1, 3).Slice(1, 1, 3),
      subtensor));
}
