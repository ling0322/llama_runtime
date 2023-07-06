#include "nn/device.h"

namespace llama {
namespace nn {

Device::Device() : _type(Type::kUnknown) {}
Device::Device(Type type) : _type(type) {}

Device Device::createForCPU() {
  return Device(Type::kCpu);
}

}
}
