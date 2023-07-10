#include "flint/util.h"

#include "flint/operators.h"
#include "llyn/error.h"

namespace flint {

void readParameters(const std::string &model_path, Module *module) {
  TensorMap state_dict;
  state_dict.read(model_path);

  module->initParameters(state_dict);
}

std::vector<Tensor> readAllTensors(const std::string &filename) {
  std::vector<Tensor> tensors;

  std::unique_ptr<ly::ReadableFile> fp = ly::ReadableFile::open(filename);
  for (; ; ) {
    Tensor A;
    try {
      A.read(fp.get());
    } catch (const ly::OutOfRangeError &) {
      break;
    }
    
    tensors.emplace_back(A);
  }

  return tensors;
}

Context getCtxForCPU() {
  auto F = Operators::create(Device::createForCPU());

  Context ctx;
  ctx.setDevice(Device::createForCPU());
  ctx.setF(std::move(F));

  return ctx;
}

}