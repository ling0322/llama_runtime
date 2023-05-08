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
class CpuOperators : public Operators {
 public:
  // create a instance of CpuOperators
  static std::unique_ptr<Operators> Create();

  // implement interface Operators
  Tensor Lookup(const Tensor &table, const Tensor &indices) override;
  Tensor MatMul(const Tensor &a, const Tensor &b) override;
  Tensor Mul(const Tensor &input, float other) override;
  Tensor Softmax(const Tensor &input) override;
  Tensor Add(const Tensor &a, const Tensor &b) override;
  Tensor Tensor_(std::initializer_list<int> shape, DType dtype) override;
  Tensor TensorLike(const Tensor &input) override;
  Tensor Rand(std::initializer_list<int> shape, DType dtype) override;
  Tensor Zeros(std::initializer_list<int> shape, DType dtype) override;
  Tensor Contiguous(const Tensor &input) override;
  bool AllClose(const Tensor &A, const Tensor &B) override;
  void Print(const Tensor &tensor) override;
  Tensor LayerNorm(
      const Tensor &input,
      const Tensor &weight,
      const Tensor &bias,
      float eps) override;

 private:
  // internal implementation of the operators.
  class Impl;
  std::unique_ptr<Impl> impl_;

  // make the constructor private
  CpuOperators() = default;
};

}  // namespace nn
}  // namespace llama

#endif  // LLM_RUNTIME_OPERATORS_CPU_H_
