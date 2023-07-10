#include "pmpack/block.h"

#include "pmpack/gemm_kernel.h"

namespace pmpack {

void QInt4Block::dequantizeTo(Block tgt) const {
  CHECK(_numRows == tgt.numRows);
  CHECK(_numCols == tgt.numCols);
  CHECK(_transposed == tgt.transposed);

  if (_transposed) {
    CHECK(_numRows % _groupSize == 0);
    int nb = _numRows / _groupSize;
    int64_t groupBytes = getGroupBytes();

    int8_t *pSrc = _data;
    float *pTgt = tgt.data;
    float *pScale = _scaleData;

    for (int c = 0; c < _numCols; ++c) {
      for (int j = 0; j < nb; ++j) {
        dequantizeInt4ToFloat32Fallback(pSrc, *pScale, _groupSize, pTgt);
        pSrc += groupBytes;
        pTgt += _groupSize;
        pScale += 1;
      }
    }
  } else {
    NOT_IMPL();
  }
}

QInt4Block::QInt4Block(
    void *data,
    float *scaleData,
    int32_t groupSize,
    int32_t numRows,
    int32_t numCols,
    bool transposed)
        : _data(reinterpret_cast<int8_t *>(data)),
          _scaleData(scaleData),
          _groupSize(groupSize),
          _numRows(numRows),
          _numCols(numCols),
          _transposed(transposed) {}

}  // namespace pmpack
