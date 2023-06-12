#ifndef LLM_RUNTIME_OPERATORS_CPU_H_
#define LLM_RUNTIME_OPERATORS_CPU_H_

#include <stdint.h>
#include <memory>
#include "nn.h"
#include "operators.h"
#include "tensor.h"

namespace llama {
namespace nn {

// the CPU implementation of Operators
class CPUOperators : public Operators {
 public:
  // create a instance of CPUOperators
  static std::unique_ptr<Operators> create();

  // implement interface Operators
  Tensor lookup(TensorCRef table, TensorCRef indices) override;
  Tensor matmul(TensorCRef a, TensorCRef b) override;
  Tensor gemv(TensorCRef A, TensorCRef x) override;
  Tensor mul(TensorCRef input, float other) override;
  Tensor softmax(TensorCRef input) override;
  Tensor gelu(TensorCRef input) override;
  Tensor add(TensorCRef a, TensorCRef b) override;
  Tensor createTensor(std::initializer_list<int> shape, DType dtype) override;
  Tensor createTensorLike(TensorCRef input) override;
  Tensor rand(std::initializer_list<int> shape, DType dtype) override;
  Tensor zeros(std::initializer_list<int> shape, DType dtype) override;
  Tensor contiguous(TensorCRef input) override;
  bool allClose(TensorCRef A, TensorCRef B) override;
  void print(TensorCRef tensor) override;
  Tensor layerNorm(TensorCRef input, TensorCRef weight, TensorCRef bias, float eps) override;
  Tensor causalMask(int max_len) override;
  Tensor cat(TensorCRef A, TensorCRef B, int dim) override;

 private:
  // internal implementation of the operators.
  class Impl;
  std::unique_ptr<Impl> _impl;

  // make the constructor private
  CPUOperators() = default;
};

}  // namespace nn
}  // namespace llama

#endif  // LLM_RUNTIME_OPERATORS_CPU_H_
