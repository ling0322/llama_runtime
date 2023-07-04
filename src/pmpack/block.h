#pragma once

#include <stdint.h>
#include "common/common.h" 

namespace llama {
namespace nn {

// block is a sub area of a matrix.
struct Block {
  float *data;
  int32_t stride;
  int32_t numRows;
  int32_t numCols;
  bool transposed;

  constexpr Block sliceRow(int row, int nr);
  constexpr Block sliceCol(int col, int nc);
  constexpr Block slice(int row, int col, int nr, int nc);
  constexpr void copyTo(Block tgt);
  constexpr Block T();
  constexpr void fillZero();
};

struct PackedBlock {
  float *data;
  int32_t packSize;
  int32_t numRows;
  int32_t numBlocks;

  constexpr Block block(int i);
};

class QInt4Block {
 public:
  QInt4Block(
      void *data,
      float *scaleData,
      int32_t groupSize,
      int32_t numRows,
      int32_t numCols,
      bool transposed);

  void dequantizeTo(Block tgt) const;
  int64_t getGroupBytes() const { return _groupSize / 2; }

 private:
  ByteType *_data;
  float *_scaleData;
  int32_t _groupSize;
  int32_t _numRows;
  int32_t _numCols;
  bool _transposed;
};

// -- class Block ----------

constexpr Block Block::sliceRow(int row, int nr) {
  return slice(row, 0, nr, numCols);
}
constexpr Block Block::sliceCol(int col, int nc) {
  return slice(0, col, numRows, nc);
}
constexpr Block Block::slice(int row, int col, int nr, int nc) {
  return Block {
    data + (transposed ? row + col * stride : row * stride + col),
    stride,
    nr,
    nc,
    transposed
  };
}

constexpr void Block::copyTo(Block tgt) {
  ASSERT(numRows == tgt.numRows);
  ASSERT(numCols == tgt.numCols);

  if ((!transposed) && (!tgt.transposed)) {
    for (int r = 0; r < numRows; ++r) {
      int tgtOffset = r * tgt.stride;
      int srcOffset = r * stride;
      for (int c = 0; c < numCols; ++c) {
        tgt.data[tgtOffset + c] = data[srcOffset + c];
      }
    }
  } else if (transposed && (!tgt.transposed)) {
    for (int r = 0; r < numRows; ++r) {
      int tgtOffset = r * tgt.stride;
      for (int c = 0; c < numCols; ++c) {
        tgt.data[tgtOffset + c] = data[r + c * stride];
      }
    }
  } else if ((!transposed) && tgt.transposed) {
    for (int r = 0; r < numRows; ++r) {
      int srcOffset = r * stride;
      for (int c = 0; c < numCols; ++c) {
        tgt.data[r + c * tgt.stride] = data[srcOffset + c];
      }
    }
  } else if (transposed && tgt.transposed) {
    for (int c = 0; c < numCols; ++c) {
      int srcOffset = c * stride;
      int tgtOffset = c * tgt.stride;
      for (int r = 0; r < numRows; ++r) {
          tgt.data[r + tgtOffset] = data[r + srcOffset];
      }
    }
  }
}

constexpr Block Block::T() {
  return Block {
    data,
    stride,
    numCols,
    numRows,
    !transposed
  };
}
constexpr void Block::fillZero() {
  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      if (transposed) {
        data[r + c * stride] = 0.0f;
      } else {
        data[r * stride + c] = 0.0f;
      }
    }
  }
}
constexpr Block PackedBlock::block(int i) {
  return Block {
    data + packSize * numRows * i,
    packSize,
    numRows,
    packSize,
    false
  };
}

}  // namespace nn
}  // namespace llama
