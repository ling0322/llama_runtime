#pragma once

#include <stdint.h>
#include <memory>
#include "pmpack/block.h"
#include "util/util.h"

namespace llama {
namespace nn {

inline PackedBlock Pack(Block src, Block buf, int pack_size) {
  int numBlock = src.numCols / pack_size;
  int kc = src.numRows;
  PackedBlock tgt { buf.data, pack_size, kc, numBlock };
  ASSERT(pack_size * numBlock * kc <= buf.numCols * buf.numRows);

  for (int b = 0; b < numBlock; ++b) {
    Block srcBlock = src.sliceCol(b * pack_size, pack_size);
    Block tgtBlock = tgt.block(b);
    srcBlock.copyTo(tgtBlock);
  }

  int nc = src.numCols % pack_size;
  if (nc) {
    Block srcBlock = src.sliceCol(numBlock * pack_size, nc);
    Block tgtBlock = tgt.block(numBlock);
    tgtBlock.fillZero();

    tgtBlock = tgtBlock.sliceCol(0, nc);
    srcBlock.copyTo(tgtBlock);
    ++tgt.numBlocks;
  }

  return tgt;
}

}  // namespace nn
}  // namespace llama
