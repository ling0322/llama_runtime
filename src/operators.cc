#include <stdlib.h>
#include <cmath>
#include <limits>
#include <memory>
#include "nn.h"
#include "operators.h"
#include "operators_cpu.h"

namespace llama {
namespace nn {

StatusOr<Operators> Operators::FromDevice(Device device) {
  switch (device.type()) {
    case Device::Type::kCpu:
      return CpuOperators::Create();
    default:
      RETURN_ABORTED() << "invalid device";
  }
}

}  // namespace nn
}  // namespace llama
