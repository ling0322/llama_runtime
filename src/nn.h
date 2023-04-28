#ifndef LLAMA_RUNTIME_NN_H_
#define LLAMA_RUNTIME_NN_H_

#include <stdint.h>
#include "common.h"
#include "operators.h"
#include "status.h"
#include "tensor.h"
#include "util.h"

namespace llama {
namespace nn {

class Function;
class Tensor;

class Device {
 public:
  enum class Type {
    kCpu,
    kUnknown
  };

  // returns a CPU device
  static Device CPU();

  // construct device by device type
  Device(Type type);

  // get type of the device
  Type type() const { return type_; }

 private:
  Type type_;
};

// string -> Tensor dictioary. Usually used to store state-dict or kv-cache
// for a neural network
class StateDict {
 public:
  Status Read(const std::string &filename);

  // get state by name. Crash if name not exist.
  Tensor &operator[](const std::string &name);

  // 

}; 

// context for a module including operator set, device info and the namespace
class Context {
 public:
  Context WithName(const std::string &name);

  std::string name(const std::string &name);

 private:
  std::string ns_;
};

// base class for all nn modules
class Module {
 public:
  // load the module states from `state_dict`
  virtual Status Load(const StateDict &state_dict) = 0;

 protected:
  Operators *F_;
  Namespace ns_;
  Device device_;
};

struct ModuleConfig {
  static constexpr int kEmptyIntProp = -1;

  // share by all modules
  Operators *F;
  Namespace ns;
  Device device;

  int d_model;
};

// linear layer in the nn.
class Linear : public Module {
 public:
  // create Linear module from config. 
  // ModuleConfig Args:
  //   d_model   
  static StatusOr<Linear> FromConfig(const ModuleConfig &config);

  // initialize the module from context
  Status Load(const StateDict &state_dict) override;

  // forward input through this module and returns the output
  Tensor Forward(const Tensor &input) const;

 private:
  Tensor w_;
  Tensor b_;

  Linear();
};

}  // namespace nn
}  // namespace llama

#endif  // LLAMA_RUNTIME_NN_H_
