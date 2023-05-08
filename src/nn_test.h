#ifndef LLAMA_RUNTIME_NN_TEST_H_
#define LLAMA_RUNTIME_NN_TEST_H_

#include <stdint.h>
#include <unordered_map>
#include "nn.h"

namespace llama {
namespace nn {

constexpr int kDModel0 = 16;
constexpr int kDModel1 = 20;
constexpr int kSeqLen = 5;
constexpr int kBatchSize = 2;

void TestSingleInputOutputModule(Context ctx,
                                 const std::string &model_path,
                                 const std::string &test_case_path,
                                 Module *layer);

}  // namespace nn
}  // namespace llama

#endif  // LLAMA_RUNTIME_NN_TEST_H_
