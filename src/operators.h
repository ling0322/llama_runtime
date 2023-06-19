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
  static std::unique_ptr<Operators> create(Device device);

  // retrieve word embeddings using indices. Input is a long tensor with indices and the output is
  // the word embeddings for these indices.
  // Args:
  //   table <float>(V, D): the embedding table. V is vocab size and D is the embedding dimension.
  //   indices <long>(N, L): the indices.
  // Returns:
  //   <float>(N, L, D): the word embedding tensor.
  virtual Tensor lookup(const Tensor &table, const Tensor &indices) = 0;

  // apply layer normalization over the last dimension of inputs.
  //   y_ij = (x_ij - E[x]) / sqrt(Var[X] + eps)
  //   y_ij = y_ij * weight_j + bias_j
  // Args:
  //   input <float>(..., D): input tensor.
  //   weight <float>(D): weight tensor.
  //   bias <float>(D): bias tensor.
  // Return:
  //   <float>(..., D): layer normalized input.
  virtual Tensor layerNorm(
      const Tensor &input, const Tensor &weight, const Tensor &bias, float eps) = 0;

  // matrix multiplication of tensor A and B. It will dispatch the operator to different routines
  // according to the input shape of A and B.
  // Args:
  //   A <float>(...): tensor A;
  //   B <float>(...): tensor B;
  // Return:
  //   <float>(<batch-dims>, M): matrix multiplication result of A and B.
  virtual Tensor matmul(const Tensor &A, const Tensor &B) = 0;

  // Element wise multiply input and other.
  virtual Tensor mul(const Tensor &input, float other) = 0;

  // Apply softmax on the last dimension of input
  virtual Tensor softmax(const Tensor &input) = 0;

  // return input + other.
  virtual Tensor add(const Tensor &input, const Tensor &other) = 0;

  // Applies the Gaussian Error Linear Units function for `input`. Here it use the approximate
  // version of GELU:
  //   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
  // Args:
  //   input <float>(..., D): input tensor.
  // Returns:
  //   <float>(..., D): outpur tensor.
  virtual Tensor gelu(const Tensor &input) = 0;

  // create a tensor with specified shape and dtype. Data in this tensor is uninitialize.
  // Args:
  //   shape: shape of the new tensor.
  //   dtype: data type of the new tensor.
  // Returns:
  //   the tensor with specified shape and dtype.
  virtual Tensor createTensor(std::initializer_list<int> shape, DType dtype) = 0;

  // returns a uninitialized tensor with the same shape and dtype as input
  virtual Tensor createTensorLike(const Tensor &input) = 0;

  // Returns a tensor filled with random numbers from a uniform distribution on
  // the interval [0, 1) 
  virtual Tensor rand(std::initializer_list<int> shape, DType dtype) = 0;

  // Returns a tensor filled with 0
  virtual Tensor zeros(std::initializer_list<int> shape, DType dtype) = 0;

  // Return a contiguous in memory tensor containing the same data as input
  virtual Tensor contiguous(const Tensor &input) = 0;

  // return true if two tensors are element-wise equal within a tolerance
  // (rtol=1e-05, atol=1e-08)
  virtual bool allClose(const Tensor &A, const Tensor &B) = 0;

  // Print the tensor to stdout,
  virtual void print(const Tensor &tensor) = 0;

  // Returns a tensor of causal mask. For the position not allowed to attend it would be +inf, for
  // the position allowed leave 0.0f.
  // Args:
  //   max_len: max length of the sequence.
  // Returns:
  //   <float>(max_len, max_len): the causal mask.
  virtual Tensor causalMask(int max_len) = 0;

  // Concat two tensors in the given dimension. Besides the dimension to concat, the two tensors
  // must have the same shape.
  // Args:
  //   A: the first tensor.
  //   B: the second tensor.
  //   dim (int): the dimension to concat alone.
  // Returns:
  //   C: concatenated tensor.
  virtual Tensor cat(const Tensor &A, const Tensor &B, int dim) = 0;
};

}  // namespace nn
}  // namespace llama

#endif  // LLM_RUNTIME_OPERATORS_H_
