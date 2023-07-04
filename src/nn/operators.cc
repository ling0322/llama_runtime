#include <stdlib.h>
#include <cmath>
#include <limits>
#include <memory>
#include "nn/nn.h"
#include "nn/operators.h"
#include "nn/operators_cpu.h"

namespace llama {
namespace nn {

std::unique_ptr<Operators> Operators::create(Device device) {
  switch (device.getType()) {
    case Device::Type::kCpu:
      return CPUOperators::create();
    default:
      throw AbortedException("invalid device");
  }
}

}  // namespace nn
}  // namespace llama
