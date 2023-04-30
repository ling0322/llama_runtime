#ifndef LLM_RUNTIME_OPERATORS_H_
#define LLM_RUNTIME_OPERATORS_H_

#include <stdint.h>
#include "nn.h"
#include "tensor.h"

namespace llama {
namespace nn {

// base functional interface to apply operators for Tensor
class Operators {
 public:
  // get instance of Function for CPU device
  static StatusOr<Operators> FromDevice(Device device);

  // virtual Tensor Lookup(const Tensor &table, const Tensor &indices) = 0;
  // virtual Tensor LayerNorm(const Tensor &input) = 0;

  // Matrix product of two tensors.
  // Args:
  //   A <float>(M, N): matrix A;
  //   B <float>(N, K): matrix B;
  // Returns:
  //   (M, K); A dot B
  virtual Tensor MatMul(const Tensor &a, const Tensor &b) = 0;

  // Element wise multiply input and other.
  virtual Tensor Mul(const Tensor &input, float other) = 0;

  // Apply softmax on the last dimension of input
  virtual Tensor Softmax(const Tensor &input) = 0;

  // return input + other.
  virtual Tensor Add(const Tensor &input, const Tensor &other) = 0;

  // create a tensor with specified shape and dtype. Data in this tensor is
  // uninitialize.
  virtual Tensor Tensor_(std::initializer_list<int> shape, DType dtype) = 0;

  // returns a uninitialized tensor with the same shape and dtype as input
  virtual Tensor TensorLike(const Tensor &input) = 0;

  // Returns a tensor filled with random numbers from a uniform distribution on
  // the interval [0, 1) 
  virtual Tensor Rand(std::initializer_list<int> shape, DType dtype) = 0;

  // Returns a tensor filled with 0
  virtual Tensor Zeros(std::initializer_list<int> shape, DType dtype) = 0;

  // Return a contiguous in memory tensor containing the same data as input
  // virtual Tensor Contiguous(const Tensor &input) = 0;

  // return true if two tensors are element-wise equal within a tolerance
  // (rtol=1e-05, atol=1e-08)
  virtual bool AllClose(const Tensor &A, const Tensor &B) = 0;

  // Print the tensor to stdout,
  virtual void Print(const Tensor &tensor) = 0;
};

}  // namespace nn
}  // namespace llama

#endif  // LLM_RUNTIME_OPERATORS_H_
