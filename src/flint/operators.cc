#include <stdlib.h>
#include <cmath>
#include <limits>
#include <memory>
#include "flint/nn.h"
#include "flint/operators.h"
#include "flint/cpu_operators.h"
#include "llyn/error.h"

namespace flint {

std::unique_ptr<Operators> Operators::create(Device device) {
  switch (device.getType()) {
    case Device::Type::kCpu:
      return CPUOperators::create();
    default:
      throw ly::AbortedError("invalid device");
  }
}

}  // namespace flint
