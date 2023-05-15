#include "test_helper.h"
#include "nn.h"

#include "nn_test_helper.h"
#include "operators.h"
#include "util.h"

using namespace llama;
using namespace nn;

// test nn module that only have one input tensor and one return tensor.
template<class TModule>
void TestSingleInOutTensorModule(Context ctx,
                                 const std::string &model_path,
                                 const std::string &test_case_path,
                                 TModule *module) {
  MustReadParameters(model_path, module);
  std::vector<Tensor> tensors = MustReadAllTensors(test_case_path);

  REQUIRE(tensors.size() % 2 == 0);
  for (int i = 0; i < tensors.size(); i += 2) {
    Tensor A = tensors[i];
    Tensor C_ref = tensors[i + 1];

    Tensor C = module->Forward(A);
    REQUIRE(ctx.F()->AllClose(C, C_ref));
  }
}

TEST_CASE("test Linear module", "[core][nn][module]") {
  util::Path model_dir = util::Path("data") / "test";
  Context ctx = MustGetCtxForCPU();

  util::Path model_path = model_dir / "linear-model.params.bin";
  util::Path tensor_file = model_dir / "linear-model.test_tensors.bin";

  StatusOr<Linear> linear = Linear::Create(ctx, kDModel0, kDModel1);
  REQUIRE(linear.ok());

  TestSingleInOutTensorModule<Linear>(
      ctx,
      model_path.string(),
      tensor_file.string(),
      linear.get());
}

TEST_CASE("test LayerNorm module", "[core][nn][module]") {
  util::Path model_dir = util::Path("data") / "test";
  Context ctx = MustGetCtxForCPU();

  util::Path model_path = model_dir / "layer-norm-model.params.bin";
  util::Path tensor_file = model_dir / "layer-norm-model.test_tensors.bin";

  StatusOr<LayerNorm> layer_norm = LayerNorm::Create(ctx, kDModel0);
  REQUIRE(layer_norm.ok());

  TestSingleInOutTensorModule<LayerNorm>(
      ctx,
      model_path.string(),
      tensor_file.string(),
      layer_norm.get());
}
