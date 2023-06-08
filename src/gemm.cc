#include "gemm.h"

#include <stdlib.h>
#include "log.h"

namespace llama {
namespace nn {

/*
struct GEMMConst {
  static constexpr int MC = 288;
  static constexpr int KC = 512;
  static constexpr int NC = 4096;
  static constexpr int MR = 6;
  static constexpr int NR = 16;
};
*/

struct GEMMConst {
  static constexpr int MC = 576;
  static constexpr int KC = 512;
  static constexpr int NC = 4096;
  static constexpr int MR = 12;
  static constexpr int NR = 32;
};

GEMM


}  // namespace nn
}  // namespace llama
