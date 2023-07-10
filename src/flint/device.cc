#include "flint/device.h"

namespace flint {

Device::Device() : _type(Type::kUnknown) {}
Device::Device(Type type) : _type(type) {}

Device Device::createForCPU() {
  return Device(Type::kCpu);
}

}  // namespace flint
