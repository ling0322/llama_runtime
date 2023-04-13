#include "common.h"

namespace llama {

template <>
DType TypeID<float>() {
  return DType::kFloat;
}

template DType TypeID<float>();

int SizeOfDType(DType dtype) {
  switch (dtype) {
    case DType::kFloat:
      return 4;
    default:
      NOT_IMPL();
  }
}

bool IsValidDType(DType dtype) {
  switch (dtype) {
    case DType::kFloat:
      return true;
    default:
      return false;
  }
}



}  // namespace llama
