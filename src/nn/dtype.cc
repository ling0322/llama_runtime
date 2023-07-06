#include "nn/dtype.h"

#include "util/log.h"

namespace llama {
namespace nn {

template <>
DType getTypeID<float>() {
  return DType::kFloat;
}
template <>
DType getTypeID<int64_t>() {
  return DType::kLong;
}
template <>
DType getTypeID<QInt4x2Fp32>() {
  return DType::kQInt4Fp32;
}

template DType getTypeID<float>();
template DType getTypeID<int64_t>();
template DType getTypeID<QInt4x2Fp32>();


template <>
DType getDequantTypeID<QInt4x2Fp32>() {
  return DType::kFloat;
}

template DType getDequantTypeID<QInt4x2Fp32>();

int64_t getDTypeTotalSize(DType dtype, int64_t numel) {
  switch (dtype) {
    case DType::kFloat:
      return 4 * numel;
    case DType::kLong:
      return 8 * numel;
    case DType::kQInt4Fp32:
      CHECK(numel % 2 == 0);
      return numel / 2;
    default:
      NOT_IMPL();
  }
}

bool isValidDType(DType dtype) {
  switch (dtype) {
    case DType::kFloat:
    case DType::kLong:
    case DType::kQInt4Fp32:
      return true;
    default:
      return false;
  }
}


}
}
