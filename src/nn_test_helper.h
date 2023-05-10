#ifndef LLMRUNTIME_NN_TEST_HELPER_H_
#define LLMRUNTIME_NN_TEST_HELPER_H_

#include <stdint.h>
#include <unordered_map>
#include "nn.h"

namespace llama {
namespace nn {

constexpr int kDModel0 = 16;
constexpr int kDModel1 = 20;
constexpr int kSeqLen = 5;
constexpr int kBatchSize = 2;
constexpr int kNumHeads = 2;

// The following functions SHOULD only been used in test code. It use the
// REQUIRE macro from catch2.

// Read parameters from model_path. ONLY for test.
void MustReadParameters(const std::string &model_path, Module *module);

// Read a list of tensors from file. 
std::vector<Tensor> MustReadAllTensors(const std::string &filename);

// Create a context for CPU device.
Context MustGetCtxForCPU();

}  // namespace nn
}  // namespace llama

#endif  // LLMRUNTIME_NN_TEST_HELPER_H_
