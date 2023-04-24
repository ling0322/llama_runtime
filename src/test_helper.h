#ifndef LLAMA_CC_TEST_COMMON_H_
#define LLAMA_CC_TEST_COMMON_H_

#include <iostream>
#include <vector>
#include "common.h"
#include "log.h"

// CHECK from log.h conflicts with catch2
#undef CHECK

#include "third_party/catch2/catch_amalgamated.hpp"

namespace llama {
namespace test_helper {

// read 16kHz, 16bit mono-channel wave file and returns samples
std::vector<float> ReadAudio(const std::string &wave_file);

}  // namespace test_helper
}  // namespace llama

#endif  // LLAMA_CC_TEST_COMMON_H_
