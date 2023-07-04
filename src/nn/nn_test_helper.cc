#include "nn/nn_test_helper.h"

#include "nn/nn.h"
#include "nn/operators.h"
#include "common/test_helper.h"

namespace llama {
namespace nn {

void MustReadParameters(const std::string &model_path, Module *module) {
  TensorMap state_dict;
  state_dict.read(model_path);

  module->initParameters(state_dict);
}

std::vector<Tensor> MustReadAllTensors(const std::string &filename) {
  std::vector<Tensor> tensors;

  std::unique_ptr<ReadableFile> fp = ReadableFile::open(filename);
  for (; ; ) {
    Tensor A;
    try {
      A.read(fp.get());
    } catch (const Exception &) {
      break;
    }
    
    tensors.emplace_back(A);
  }

  return tensors;
}

Context MustGetCtxForCPU() {
  auto F = Operators::create(Device::createForCPU());

  Context ctx;
  ctx.setDevice(Device::createForCPU());
  ctx.setF(std::move(F));

  return ctx;
}

}  // namespace nn
}  // namespace llama
