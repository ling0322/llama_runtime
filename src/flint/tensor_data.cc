#include "flint/tensor_data.h"

#include <stdint.h>
#include <memory>

#include "flint/device.h"
#include "flint/dtype.h"
#include "llyn/platform.h"

namespace flint {


// -- class TensorData ---------------------------------------------------------

TensorData::TensorData()
    : _data(nullptr),
      _scaleData(nullptr),
      _groupSize(0),
      _numel(0),
      _dtype(DType::kUnknown) {}

// -- class CPUTensorData ---------------------------------------------------------

class CPUTensorData : public TensorData {
 public:
  ~CPUTensorData();

  Device getDevice() const override;
};

std::shared_ptr<TensorData> TensorData::create(int64_t numel, DType dtype) {
  auto tensorData = std::make_shared<CPUTensorData>();
  int64_t size = getDTypeTotalSize(dtype, numel);
  tensorData->_data = (int8_t *)ly::alloc32ByteAlignedMem(size);
  tensorData->_numel = numel;
  tensorData->_dtype = dtype;

  return tensorData;
}

CPUTensorData::~CPUTensorData() {
  if (_data) {
    ly::free32ByteAlignedMem(_data);
    _data = nullptr;
  }

  if (_scaleData) {
    ly::free32ByteAlignedMem(_scaleData);
    _scaleData = nullptr;
  }
}

Device CPUTensorData::getDevice() const {
  return Device(Device::Type::kCpu);
}

}  // namespace flint
