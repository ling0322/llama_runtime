#include "test_helper.h"
#include "nn.h"
#include "operators.h"
#include "util.h"

using namespace llama;
using namespace nn;

TEST_CASE("test Linear module", "[core][nn][module]") {
  TensorDict state_dict;
  util::Path model_dir = util::Path("data") / "test";
  util::Path model_path = model_dir / "linear-model.params.bin";
  Status status = state_dict.Read(model_path.string());
  REQUIRE(status.ok());

  StatusOr<Operators> F = Operators::FromDevice(Device::CPU());
  REQUIRE(F.ok());

  Context ctx;
  ctx.set_device(Device::CPU());
  ctx.set_F(F.get());

  StatusOr<Linear> linear = Linear::Create(ctx, 16, 20);
  REQUIRE(linear.ok());

  status = linear->InitParameters(state_dict);
  REQUIRE(status.ok());

  util::Path tensor_file = model_dir / "linear-model.test_tensors.bin";
  StatusOr<ReadableFile> fp = ReadableFile::Open(tensor_file.string());
  REQUIRE(fp.ok());

  Tensor A, C, C_ref;

  for (; ; ) {
    status = A.Read(fp.get());
    if (IsOutOfRange(status)) {
      // EOF reached
      break;
    }
    REQUIRE(status.ok());

    status = C_ref.Read(fp.get());
    REQUIRE(status.ok());

    C = linear->Forward(A);
    REQUIRE(F->AllClose(C, C_ref));
  }
}
