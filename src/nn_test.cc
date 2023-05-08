#include "test_helper.h"
#include "nn.h"
#include "operators.h"
#include "util.h"

namespace llama {
namespace nn {

constexpr int kDModel0 = 16;
constexpr int kDModel1 = 20;

// test nn module that only have one input value and one return value.
void TestSingleInputOutputModule(Context ctx,
                                 const std::string &model_path,
                                 const std::string &test_case_path,
                                 Module *layer) {
  TensorDict state_dict;
  Status status = state_dict.Read(model_path);
  REQUIRE(status.ok());

  status = layer->InitParameters(state_dict);
  REQUIRE(status.ok());

  StatusOr<ReadableFile> fp = ReadableFile::Open(test_case_path);
  REQUIRE(fp.ok());

  Tensor A, C, C_ref;

  Operators *F = ctx.F();
  for (; ; ) {
    status = A.Read(fp.get());
    if (IsOutOfRange(status)) {
      // EOF reached
      break;
    }
    REQUIRE(status.ok());

    status = C_ref.Read(fp.get());
    REQUIRE(status.ok());

    C = layer->Forward(A);
    REQUIRE(F->AllClose(C, C_ref));
  }
}

}  // namespace nn
}  // namespace llama

using namespace llama;
using namespace nn;

TEST_CASE("test Linear module", "[core][nn][module]") {
  util::Path model_dir = util::Path("data") / "test";
  StatusOr<Operators> F = Operators::FromDevice(Device::CPU());
  REQUIRE(F.ok());

  Context ctx;
  ctx.set_device(Device::CPU());
  ctx.set_F(F.get());

  // linear
  util::Path model_path = model_dir / "linear-model.params.bin";
  util::Path tensor_file = model_dir / "linear-model.test_tensors.bin";

  StatusOr<Linear> linear = Linear::Create(ctx, kDModel0, kDModel1);
  REQUIRE(linear.ok());

  TestSingleInputOutputModule(
      ctx,
      model_path.string(),
      tensor_file.string(),
      linear.get());

  // layer-norm
  model_path = model_dir / "layer-norm-model.params.bin";
  tensor_file = model_dir / "layer-norm-model.test_tensors.bin";

  StatusOr<LayerNorm> layer_norm = LayerNorm::Create(ctx, kDModel0);
  REQUIRE(linear.ok());

  TestSingleInputOutputModule(
      ctx,
      model_path.string(),
      tensor_file.string(),
      layer_norm.get());
}
