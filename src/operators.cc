#include <stdlib.h>
#include <cmath>
#include <limits>
#include <memory>
#include "nn.h"
#include "operators.h"

namespace llama {
namespace nn {

// create the instance of CpuOperators
std::unique_ptr<Operators> CreateCpuOperators();

StatusOr<Operators> Operators::FromDevice(Device device) {
  switch (device.type()) {
    case Device::Type::kCpu:
      return CreateCpuOperators();
    default:
      RETURN_ABORTED() << "invalid device";
  }
}

}  // namespace nn
}  // namespace llama
