#include "common/test_helper.h"
#include "flint/tensor.h"

#include "flint/nn_test_helper.h"
#include "flint/operators.h"
#include "util/util.h"

using namespace llama;
using namespace nn;

TEST_CASE("test subtensor and slice", "[core][nn][tensor]") {
  Context ctx = MustGetCtxForCPU();
  Tensor tensor = Tensor::create<float>({4, 4}, {
    0.0f, 0.1f, 0.2f, 0.3f,
    0.4f, 0.5f, 0.6f, 0.7f,
    0.8f, 0.9f, 1.0f, 1.1f,
    1.2f, 1.3f, 1.4f, 1.5f
  });

  // slice (dim 0)
  Tensor subtensor = Tensor::create<float>({2, 4}, {
    0.4f, 0.5f, 0.6f, 0.7f,
    0.8f, 0.9f, 1.0f, 1.1f,
  });
  REQUIRE(ctx.F()->allClose(tensor.slice(1, 3), subtensor));
  
  // subtensor
  subtensor = Tensor::create<float>({4}, {
    0.4f, 0.5f, 0.6f, 0.7f,
  });
  REQUIRE(ctx.F()->allClose(tensor.subtensor(1), subtensor));

  // slice (any dim)
  subtensor = Tensor::create<float>({2, 2}, {
    0.5f, 0.6f,
    0.9f, 1.0f,
  });
  REQUIRE(ctx.F()->allClose(tensor.slice(0, 1, 3).slice(1, 1, 3), subtensor));
}
