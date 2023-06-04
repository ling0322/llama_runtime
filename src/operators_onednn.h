#ifndef FASTALPACA_OPERATORS_ONEDNN_H_
#define FASTALPACA_OPERATORS_ONEDNN_H_

#include <stdint.h>
#include <memory>
#include "nn.h"
#include "operators.h"
#include "operators_cpu.h"
#include "tensor.h"

namespace llama {
namespace nn {

// the oneDNN (CPU) implementation of Operators
class OneDnnOperators : public CpuOperators {
 public:
  // create a instance of CpuOperators
  static std::unique_ptr<Operators> Create();

  // override CPU operators
  Tensor MatMul(const Tensor &a, const Tensor &b) override;

 private:
  // internal implementation of the operators.
  class Impl;
  std::unique_ptr<Impl> impl_;

  // make the constructor private
  OneDnnOperators() = default;
};

}  // namespace nn
}  // namespace llama

#endif  // FASTALPACA_OPERATORS_DNNL_H_
