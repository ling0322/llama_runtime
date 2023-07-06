#pragma once

#include <stdint.h>

namespace llama {
namespace nn {

struct QInt4x2Fp32 {
  uint8_t v0 : 4;
  uint8_t v1 : 4;
};
static_assert(sizeof(QInt4x2Fp32) == 1);

typedef int64_t LongType;

enum class DType : int16_t { 
  kUnknown = 0,
  kFloat = 1,
  kLong = 2,
  kQInt4Fp32 = 3
};

// get type-id
template <typename T>
DType getTypeID();

template <typename T>
DType getDequantTypeID();

// get the total size of specific number of elements with dtype.
int64_t getDTypeTotalSize(DType dtype, int64_t numel);

// return true of DType is valid
bool isValidDType(DType dtype);

}
}