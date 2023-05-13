#ifndef LLAMA_CC_TEST_COMMON_H_
#define LLAMA_CC_TEST_COMMON_H_

#include <iostream>
#include <vector>
#include "common.h"
#include "log.h"

// CHECK from log.h conflicts with catch2
#undef CHECK

#include "../third_party/catch2/catch_amalgamated.hpp"

#endif  // LLAMA_CC_TEST_COMMON_H_
