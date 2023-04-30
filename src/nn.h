#ifndef LLAMA_RUNTIME_NN_H_
#define LLAMA_RUNTIME_NN_H_

#include <stdint.h>
#include <unordered_map>
#include "common.h"
#include "status.h"
#include "tensor.h"
#include "util.h"

namespace llama {
namespace nn {

class Operators;
class Function;
class Tensor;

class Device {
 public:
  enum class Type {
    kCpu,
    kUnknown
  };

  static Device CPU();

  // construct device by device type
  Device();
  Device(Type type);

  // get type of the device
  Type type() const { return type_; }

 private:
  Type type_;
};

// string -> Tensor dictioary. Usually used to store state-dict or kv-cache
// for a neural network
class TensorDict {
 public:
  Status Read(const std::string &filename);

  // get tensor by name. abort if not exist.
  Tensor &operator[](const std::string &name);

  // get tensor by name. return abort if not exist.
  Status get(const std::string &name, Tensor *tensor) const;

  // return true if the tensor exists. 
  bool has_tensor(const std::string &name) const;

 private:
  std::unordered_map<std::string, Tensor> dict_;
}; 

// context for a module including operator set, device info and the namespace
class Context {
 public:
  // default constructor (root context).
  Context();

  // return a copy of this context with a new name under current context
  // namespace
  Context WithName(const std::string &name) const;

  // get a tensor or module name under this context. If no parameter given,
  // return the name of the context itself
  std::string name(const std::string &name) const;
  std::string name() const { return ns_; }

  // operator set
  Operators *F() const { return F_; }
  void set_F(Operators *F) { F_ = F; }

  // device.
  const Device &device() const; 
  void set_device(const Device &device) { device_ = device; }

 private:
  std::string ns_;
  Operators *F_;
  Device device_;
};

// base class for all nn modules
class Module {
 public:
  // load the module states from `state_dict`
  virtual Status InitParameters(const TensorDict &state_dict) = 0;

  // get context of the module.
  const Context &ctx() const { return ctx_; }

 protected:
  Context ctx_;
};

// linear layer in the nn.
class Linear : public Module {
 public:
  // create Linear module from context. 
  static StatusOr<Linear> Create(const Context &ctx, int d_model);

  // initialize the module from context
  Status InitParameters(const TensorDict &state_dict) override;

  // forward input through this module and returns the output
  Tensor Forward(const Tensor &input) const;

 private:
  // tensor names.
  static constexpr char kWeight[] = "weight";
  static constexpr char kBias[] = "bias";

  Tensor w_;
  Tensor b_;

  int d_model_;

  Linear();
};

}  // namespace nn
}  // namespace llama

#endif  // LLAMA_RUNTIME_NN_H_
