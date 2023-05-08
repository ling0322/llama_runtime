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

  // retrieve word embeddings using indices. Input is a long tensor with indices
  // and the output is the word embeddings for these indices.
  // Args:
  //   table <float>(V, D): the embedding table. V is vocab size and D is the
  //                        embedding dimension.
  //   indices <long>(N, L): the indices.
  // Returns:
  //   <float>(N, L, D): the word embedding tensor.
  virtual Tensor Lookup(const Tensor &table, const Tensor &indices) = 0;

  // apply layer normalization over the last dimension of inputs.
  //   y_ij = (x_ij - E[x]) / sqrt(Var[X] + eps)
  //   y_ij = y_ij * weight_j + bias_j
  // Args:
  //   input <float>(..., D): input tensor.
  //   weight <float>(D): weight tensor.
  //   bias <float>(D): bias tensor.
  // Return:
  //   <float>(..., D): layer normalized input.
  virtual Tensor LayerNorm(
      const Tensor &input,
      const Tensor &weight,
      const Tensor &bias,
      float eps) = 0;

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
  // Args:
  //   shape: shape of the new tensor.
  //   dtype: data type of the new tensor.
  // Returns:
  //   the tensor with specified shape and dtype.
  virtual Tensor Tensor_(std::initializer_list<int> shape, DType dtype) = 0;

  // returns a uninitialized tensor with the same shape and dtype as input
  virtual Tensor TensorLike(const Tensor &input) = 0;

  // Returns a tensor filled with random numbers from a uniform distribution on
  // the interval [0, 1) 
  virtual Tensor Rand(std::initializer_list<int> shape, DType dtype) = 0;

  // Returns a tensor filled with 0
  virtual Tensor Zeros(std::initializer_list<int> shape, DType dtype) = 0;

  // Return a contiguous in memory tensor containing the same data as input
  virtual Tensor Contiguous(const Tensor &input) = 0;

  // return true if two tensors are element-wise equal within a tolerance
  // (rtol=1e-05, atol=1e-08)
  virtual bool AllClose(const Tensor &A, const Tensor &B) = 0;

  // Print the tensor to stdout,
  virtual void Print(const Tensor &tensor) = 0;
};

}  // namespace nn
}  // namespace llama

#endif  // LLM_RUNTIME_OPERATORS_H_
