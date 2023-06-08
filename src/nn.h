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

// the config section name in ini file.
constexpr char kConfigSection[] = "config";
constexpr char kModelSection[] = "model";
constexpr char kParamsSection[] = "params";

constexpr double kPi = 3.14159265359;

class Operators;
class Function;
class Tensor;

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

// string -> Tensor dictioary. Usually used to store state-dict or kv-cache for a neural network.
class TensorMap {
 public:
  void read(const std::string &filename);

  // get tensor by name. abort if not exist.
  Tensor getTensor(const std::string &name) const;

  // put tensor.
  void putTensor(const std::string &name, TensorCRef tensor);

  // try to get tensor by name. return AbortError() if not exist.
  bool getTensorNoThrow(const std::string &name, Tensor *tensor) const;

  // return true if the tensor exists. 
  bool hasTensor(const std::string &name) const;

 private:
  std::unordered_map<std::string, Tensor> _dict;
}; 

// context for a module including operator set, device info and the namespace
class Context {
 public:
  // default constructor (root context).
  Context();

  // return a copy of this context with a new name under current context namespace.
  Context withName(const std::string &name) const;

  // get a tensor or module name under this context. If no parameter given, return the name of the
  // context itself
  std::string name(const std::string &name) const;
  std::string name() const { return _ns; }

  // operator set
  Operators *F() const { return _F.get(); }
  void setF(std::shared_ptr<Operators> F) { _F = F; }

  // device.
  const Device &getDevice() const; 
  void setDevice(const Device &device) { _device = device; }

 private:
  std::string _ns;
  std::shared_ptr<Operators> _F;
  Device _device;
};

// base class for all nn modules.
class Module {
 public:
  // load the module states from `state_dict`
  virtual void initParameters(const TensorMap &stateDict) = 0;
};

// base class for language model.
class LanguageModel {
 public:
  // Forward input token ids through this language model. It will update the `past` state and
  // return the hidden state of last layer.
  // Args:
  //   past (TensorMap): key-value cache.
  //   inputs <long>(N, L): prompt token ids.
  // Returns:
  //   <float>(N, L, D): hidden state from last layer.
  virtual Tensor forward(TensorMap *past, TensorCRef inputs) const = 0;

  // Forward the hidden state from last layer and get the logits. hidden_state is usually the
  // return value of Forward().
  // Args:
  //   hidden_state <float>(N, L, D): hidden state from last layer.
  // Returns:
  //   <float>(N, L, V): logits. V is vocabulary size.
  virtual Tensor logits(TensorCRef hiddenState) const = 0;
};

// linear layer.
class Linear : public Module {
 public:
  // create Linear module from context. 
  static std::unique_ptr<Linear> create(const Context &ctx, int in_features, int out_features);

  // initialize the module from context
  void initParameters(const TensorMap &state_dict) override;

  // forward input and return the output.
  Tensor forward(const Tensor &input) const;

 private:
  // tensor names.
  static constexpr char kWeight[] = "weight";
  static constexpr char kBias[] = "bias";

  Context _ctx;

  Tensor _w;
  Tensor _b;

  int _inFeatures;
  int _outFeatures;

  Linear();
};

// layer-norm layer.
class LayerNorm : public Module {
 public:
  static std::unique_ptr<LayerNorm> create(const Context &ctx, int d_model, float eps = 1e-5);
  
  // initialize the module from context
  void initParameters(const TensorMap &state_dict) override;

  // forward input and return the output.
  Tensor forward(const Tensor &input) const;
 
 private:
  // tensor names.
  static constexpr char kWeight[] = "weight";
  static constexpr char kBias[] = "bias";

  Context _ctx;

  Tensor _w;
  Tensor _b;

  int _dModel;
  float _eps;

  LayerNorm();
};

}  // namespace nn
}  // namespace llama

#endif  // LLAMA_RUNTIME_NN_H_
