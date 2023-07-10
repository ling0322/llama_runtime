#pragma once 

#include <stdint.h>
#include <memory>

#include "flint/device.h"
#include "flint/dtype.h"
#include "llyn/log.h"

namespace flint {

// contains dimension and stride information for an axis in tensor
class TensorData {
 public:
  static std::shared_ptr<TensorData> create(int64_t numel, DType dtype);
  static std::shared_ptr<TensorData> create(int64_t numel, int32_t groupSize, DType dtype);

  virtual ~TensorData() = default;

  // get the device of tensor data.
  virtual Device getDevice() const = 0;

  int8_t *getData() const { return _data; }

  template<typename T>
  T *getScaleData() const;

  DType getDType() const { return _dtype; }
  int64_t getSizeInBytes() const { return getDTypeTotalSize(_dtype, _numel); }
  int32_t getGroupSize() const { return _groupSize; }

 protected:
  int8_t *_data;
  int8_t *_scaleData;  // used in quantized tensor only. Scale for the quantization group.
  int64_t _numel;
  int32_t _groupSize;    // number of elements per quantization group. 0 for no quantization.
  DType _dtype;

  TensorData();
};

template<typename T>
T *TensorData::getScaleData() const  {
  CHECK(getDequantType(_dtype) == getTypeID<T>());
  return reinterpret_cast<T *>(_scaleData);
}

}  // namespace flint
