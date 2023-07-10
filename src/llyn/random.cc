#include "llyn/random.h"

namespace ly {

Random::Random() {
  uint64_t seed = static_cast<uint64_t>(time(nullptr));
  _x = seed % RandMax;
}

Random::Random(uint64_t seed) : _x(seed % RandMax) {}

void Random::fill(Span<float> l, float min, float max) {
  for (float &v : l) {
    v = static_cast<float>(static_cast<double>(nextInt()) / RandMax);
    v = min + (max - min) * v;
  }
}

void Random::fill(Span<float> l) {
  return fill(l, 0.0f, 1.0f);
}

void Random::fillUInt8(Span<uint8_t> l) {
  for (uint8_t &v : l) {
    v = nextInt() % 256;
  }
}

int32_t Random::nextInt() {
  _x = (48271 * _x) % RandMax;
  return static_cast<int32_t>(_x);
}

}  // namespace ly
