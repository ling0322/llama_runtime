#include "nn_test_helper.h"

#include "nn.h"
#include "operators.h"
#include "test_helper.h"

namespace llama {
namespace nn {

void MustReadParameters(const std::string &model_path, Module *module) {
  TensorMap state_dict;
  Status status = state_dict.Read(model_path);
  REQUIRE_OK(status);

  status = module->InitParameters(state_dict);
  REQUIRE_OK(status);
}

std::vector<Tensor> MustReadAllTensors(const std::string &filename) {
  std::vector<Tensor> tensors;

  expected_ptr<ReadableFile> fp = ReadableFile::Open(filename);
  REQUIRE(fp.ok());

  for (; ; ) {
    Tensor A;
    Status status = A.Read(fp.get());
    if (IsOutOfRange(status)) {
      // EOF reached
      break;
    }
    REQUIRE(status.ok());
    tensors.emplace_back(A);
  }

  return tensors;
}

Context MustGetCtxForCPU() {
  expected_ptr<Operators> F = Operators::FromDevice(Device::CPU());
  REQUIRE(F.ok());

  Context ctx;
  ctx.set_device(Device::CPU());
  ctx.set_F(std::move(F).shared_ptr());

  return ctx;
}

}  // namespace nn
}  // namespace llama
