#pragma once

#include <stdint.h>
#include "llyn/span.h"

namespace ly {

// random number generator.
class Random {
 public:
  static constexpr int32_t RandMax = 2147483647;  // 2^31-1

  // initialize the random number generator by current time.
  Random();
  Random(uint64_t seed);

  // fill `l` with a list of float numbers in range [0, 1) or [min, max).
  void fill(Span<float> l);
  void fill(Span<float> l, float min, float max);

  // fill `l` with a list of uint8_t numbers in range [0, 255).
  void fillUInt8(Span<uint8_t> l);

  // return next random int number in range [0, RandMax).
  int32_t nextInt();

 private:
  uint64_t _x;
};

}  // namespace ly
