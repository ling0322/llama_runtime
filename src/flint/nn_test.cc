#include "catch2/catch_amalgamated.hpp"
#include "flint/nn.h"
#include "flint/operators.h"
#include "flint/util.h"
#include "llyn/path.h"

using namespace flint;
using namespace ly;

constexpr int kDModel0 = 16;
constexpr int kDModel1 = 20;
constexpr int kSeqLen = 10;
constexpr int kBatchSize = 2;
constexpr int kNumHeads = 2;

// test nn module that only have one input tensor and one return tensor.
template<class TModule>
void TestSingleInOutTensorModule(Context ctx,
                                 const std::string &model_path,
                                 const std::string &test_case_path,
                                 TModule *module) {
  readParameters(model_path, module);
  std::vector<Tensor> tensors = readAllTensors(test_case_path);

  REQUIRE(tensors.size() % 2 == 0);
  for (int i = 0; i < tensors.size(); i += 2) {
    Tensor A = tensors[i];
    Tensor C_ref = tensors[i + 1];

    Tensor C = module->forward(A);
    REQUIRE(ctx.F()->allClose(C, C_ref));
  }
}

TEST_CASE("test Linear module", "[core][nn][module]") {
  ly::Path model_dir = ly::Path("data") / "test";
  Context ctx = getCtxForCPU();

  ly::Path model_path = model_dir / "linear-model.params.bin";
  ly::Path tensor_file = model_dir / "linear-model.test_tensors.bin";

  auto linear = Linear::create(ctx, kDModel0, kDModel1);
  TestSingleInOutTensorModule<Linear>(
      ctx,
      model_path.string(),
      tensor_file.string(),
      linear.get());
}

TEST_CASE("test LayerNorm module", "[core][nn][module]") {
  ly::Path model_dir = ly::Path("data") / "test";
  Context ctx = getCtxForCPU();

  ly::Path model_path = model_dir / "layer-norm-model.params.bin";
  ly::Path tensor_file = model_dir / "layer-norm-model.test_tensors.bin";

  auto layer_norm = LayerNorm::create(ctx, kDModel0);
  TestSingleInOutTensorModule<LayerNorm>(
      ctx,
      model_path.string(),
      tensor_file.string(),
      layer_norm.get());
}
