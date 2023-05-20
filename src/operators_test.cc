#include "test_helper.h"

#include "nn_test_helper.h"
#include "operators.h"
#include "util.h"

using namespace llama;
using namespace nn;

TEST_CASE("test softmax", "[core][nn][operators]") {
  Context ctx = MustGetCtxForCPU();

  Tensor input = Tensor::FromData<float>({3}, {0.1f, 0.2f, 0.3f});
  Tensor output = Tensor::FromData<float>({3}, {0.3006f, 0.3322f, 0.3672f});
  REQUIRE(ctx.F()->AllClose(ctx.F()->Softmax(input), output));

  constexpr float inf = std::numeric_limits<float>::infinity();
  input = Tensor::FromData<float>({3}, {0.1f, 0.2f, -inf});
  output = Tensor::FromData<float>({3}, {0.4750f, 0.5250f, 0.0f});
  REQUIRE(ctx.F()->AllClose(ctx.F()->Softmax(input), output));
}

TEST_CASE("test embedding lookup", "[core][nn][operators]") {
  Context ctx = MustGetCtxForCPU();

  Tensor wte = Tensor::FromData<float>({5, 2}, {
      0.1f, 0.2f,
      0.3f, 0.4f,
      0.2f, 0.3f,
      0.4f, 0.5f,
      0.7f, 0.8f,
  });
  Tensor input = Tensor::FromData<LongType>({2, 3}, {
      0, 1, 2,
      1, 3, 4,
  });
  Tensor output = Tensor::FromData<float>({2, 3, 2}, {
      0.1f, 0.2f,
      0.3f, 0.4f,
      0.2f, 0.3f,

      0.3f, 0.4f,
      0.4f, 0.5f,
      0.7f, 0.8f,
  });
  REQUIRE(ctx.F()->AllClose(ctx.F()->Lookup(wte, input), output));
}

