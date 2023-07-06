#include "nn/tensor_data.h"

#include <stdint.h>
#include <memory>

#include "common/common.h"
#include "nn/device.h"
#include "nn/dtype.h"
#include "util/util.h"

namespace llama {
namespace nn {


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
  tensorData->_data = (ByteType *)util::alloc32ByteAlignedMem(size);
  tensorData->_numel = numel;
  tensorData->_dtype = dtype;

  return tensorData;
}

CPUTensorData::~CPUTensorData() {
  if (_data) {
    util::free32ByteAlignedMem(_data);
    _data = nullptr;
  }

  if (_scaleData) {
    util::free32ByteAlignedMem(_scaleData);
    _scaleData = nullptr;
  }
}

Device CPUTensorData::getDevice() const {
  return Device(Device::Type::kCpu);
}

}
}
