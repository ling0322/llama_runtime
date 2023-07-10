#pragma once

namespace flint {

// storage device for tensor data.
class Device {
 public:
  enum class Type {
    kCpu,
    kUnknown
  };

  static Device createForCPU();

  // construct device by device type
  Device();
  Device(Type type);

  // get type of the device
  Type getType() const { return _type; }

 private:
  Type _type;
};

}
