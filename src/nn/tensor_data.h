#pragma once 

#include <stdint.h>
#include <memory>

#include "common/common.h"
#include "nn/device.h"
#include "nn/dtype.h"

namespace llama {
namespace nn {

// contains dimension and stride information for an axis in tensor
class TensorData {
 public:
  static std::shared_ptr<TensorData> create(int64_t numel, DType dtype);
  static std::shared_ptr<TensorData> create(int64_t numel, int32_t groupSize, DType dtype);

  virtual ~TensorData() = default;

  // get the device of tensor data.
  virtual Device getDevice() const = 0;

  ByteType *getData() const { return _data; }

  template<typename T>
  T *getScaleData() const;

  DType getDType() const { return _dtype; }
  int64_t getSizeInBytes() const { return getDTypeTotalSize(_dtype, _numel); }
  int64_t getGroupSize() const { return _groupSize; }

 protected:
  ByteType *_data;
  ByteType *_scaleData;  // used in quantized tensor only. Scale for the quantization group.
  int64_t _groupSize;    // number of elements per quantization group. 0 for no quantization.
  int64_t _numel;
  DType _dtype;

  TensorData();
};

template<typename T>
T *TensorData::getScaleData() const  {
  CHECK(getDequantTypeID<T>() == DType::kFloat);
  return reinterpret_cast<T *>(_scaleData);
}

}
}
