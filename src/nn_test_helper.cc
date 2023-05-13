#include "nn_test_helper.h"

#include "nn.h"
#include "operators.h"
#include "test_helper.h"

namespace llama {
namespace nn {

void MustReadParameters(const std::string &model_path, Module *module) {
  TensorMap state_dict;
  Status status = state_dict.Read(model_path);
  puts(status.what().c_str());
  REQUIRE(status.ok());

  status = module->InitParameters(state_dict);
  REQUIRE(status.ok());
}

std::vector<Tensor> MustReadAllTensors(const std::string &filename) {
  std::vector<Tensor> tensors;

  StatusOr<ReadableFile> fp = ReadableFile::Open(filename);
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
  StatusOr<Operators> F = Operators::FromDevice(Device::CPU());
  REQUIRE(F.ok());

  Context ctx;
  ctx.set_device(Device::CPU());
  ctx.set_F(std::move(F).shared_ptr());

  return ctx;
}

Tensor MakeTensor(Operators *F,
                  std::initializer_list<int> shape,
                  std::initializer_list<float> data) {
  Tensor tensor = F->Tensor_(shape, DType::kFloat);

  CHECK(tensor.numel() == data.size());
  std::copy(data.begin(), data.end(), tensor.data<float>());

  return tensor;
}

}  // namespace nn
}  // namespace llama
