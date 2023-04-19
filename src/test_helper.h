#ifndef LLAMA_CC_TEST_COMMON_H_
#define LLAMA_CC_TEST_COMMON_H_

#include <iostream>
#include <vector>
#include "common.h"
#include "tensor.h"
#include "log.h"

// CHECK from log.h conflicts with catch2
#undef CHECK

#include "third_party/catch2/catch_amalgamated.hpp"

namespace llama {
namespace test_helper {

// return true if 2 TensorView is near
bool IsNear(const TensorView2Df &a, const TensorView2Df &b, float eps);
bool IsNear(const TensorView &a, const TensorView &b, float eps);

// read 16kHz, 16bit mono-channel wave file and returns samples
std::vector<float> ReadAudio(const std::string &wave_file);

// read a 2D tensor from file
template<typename T>
Tensor2D<T> ReadTensor2D(const std::string &filename);

}  // namespace test_helper
}  // namespace llama

#endif  // LLAMA_CC_TEST_COMMON_H_
