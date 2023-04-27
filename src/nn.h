#ifndef LLAMA_RUNTIME_NN_H_
#define LLAMA_RUNTIME_NN_H_

#include <stdint.h>
#include "common.h"
#include "status.h"
#include "tensor.h"
#include "util.h"

namespace llama {
namespace nn {

class Function;
class Tensor;

enum class Device {
  kCpu
};

// context for recurrent neural network or auto-regressive decoder 
class Context {
 public:
  const Tensor &Get(const std::string &name);
  void Set(const std::string &name, const Tensor &value);
}; 

class Namespace {
 public:
  Namespace Sub(const std::string &sub_name);
  std::string Name(const std::string &name);

 private:
  std::string ns_;
};

// base class for all nn modules
class Module {
 public:
  // initialize the module from context
  virtual Status InitFromContext(Context *ctx) = 0;
};

// linear layer in the nn.
class Linear : public Module {
 public:
  Linear(Namespace ns, int hidden_size);

  // initialize the module from context
  Status InitFromContext(Context *ctx) override;

  // forward input through this module and returns the output
  Tensor Forward(const Tensor &input) const;

 private:
  Namespace ns_;

  Tensor w_;
  Tensor b_;
};

class Function {
 public:
  Tensor Lookup(const Tensor &table, const Tensor &indices);
  Tensor LayerNorm(const Tensor &input);

  // Matrix product of two tensors.
  // Args:
  //   A <float>(M, N): matrix A;
  //   B <float>(N, K): matrix B;
  // Returns:
  //   (M, K); A dot B
  Tensor MatMul(const Tensor &a, const Tensor &b);

  // Element wise multiply input and other.
  Tensor Mul(const Tensor &input, float other);

  // Apply softmax on the last dimension of input
  // Args:
  //   input <float>(..., L): input tensor
  // Returns:
  //   <float>(..., L): output tensor 
  Tensor Softmax(const Tensor &input);
  Tensor Add(const Tensor &a, const Tensor &b);

  // create a tensor with specified shape and dtype. Data in this tensor is
  // uninitialize.
  Tensor Tensor_(std::initializer_list<int> shape, DType dtype);

  // Returns a tensor filled with random numbers from a uniform distribution on
  // the interval [0, 1) 
  Tensor Rand(std::initializer_list<int> shape, DType dtype);

  // Returns a tensor filled with 0
  Tensor Zeros(std::initializer_list<int> shape, DType dtype);

  // Return a contiguous in memory tensor containing the same data as input
  Tensor Contiguous(const Tensor &input);

  // Print the tensor to stdout,
  void Print(const Tensor &tensor);
};

}  // namespace nn
}  // namespace llama

#endif  // LLAMA_RUNTIME_NN_H_
