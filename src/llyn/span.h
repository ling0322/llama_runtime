#pragma once

#include "llyn/internal/base_array.h"
#include "llyn/attributes.h"
#include "llyn/fixed_array.h"
#include "llyn/log.h"

namespace ly {

template<typename T>
class Span : public internal::BaseArray<T> {
 public:
  Span() noexcept : internal::BaseArray<T>() {}
  Span(T *ptr, size_type size) : internal::BaseArray<T>(ptr, size) {}

  // automatic convert initializer_list to Span<const T>.
  // NOTE: initializer_list should outlives span when using this constructor.
  // Examples:
  //   Span<const int> v = {1, 2, 3};  // WRONG: lifetime of initializer_list is shorter than v;
  template <typename U = T,
            typename = std::enable_if<std::is_const<T>::value, U>::type>
  Span(std::initializer_list<value_type> v LY_LIFETIME_BOUND) noexcept
      : Span(v.begin(), v.size()) {}

  // automatic convert std::vector<T> to Span<const T>.
  // NOTE: initializer_list should outlives span when using this constructor.
  // Examples:
  //   Span<const int> v = {1, 2, 3};  // WRONG: lifetime of initializer_list is shorter than v;
  template <typename U = T,
            typename = std::enable_if<std::is_const<T>::value, U>::type>
  Span(std::vector<value_type> &v LY_LIFETIME_BOUND) noexcept
      : Span(v.data(), v.size()) {}

  Span<T> subspan(size_type pos = 0, size_type len = npos) const {
    CHECK(pos <= size());
    len = std::min(size() - pos, len);
    return Span<T>(data() + pos, len);
  }
};

template<typename T>
constexpr Span<T> makeSpan(
    T *ptr,
    typename Span<T>::size_type size) {
  return Span<T>(ptr, size);
}
template<typename T>
constexpr Span<const T> makeConstSpan(
    const T *ptr,
    typename Span<T>::size_type size) {
  return Span<const T>(ptr, size);
}
template<typename T>
constexpr Span<T> makeSpan(std::vector<T> &v) {
  return Span<T>(v.data(), v.size());
}
template<typename T>
constexpr Span<const T> makeConstSpan(const std::vector<T> &v) {
  return Span<const T>(v.data(), v.size());
}

template<typename T>
constexpr Span<T> makeSpan(const FixedArray<T> &v) {
  return Span<T>(v.data(), v.size());
}
template<typename T>
constexpr Span<const T> makeConstSpan(const FixedArray<T> &v) {
  return Span<const T>(v.data(), v.size());
}
template<typename T>
constexpr Span<const T> makeConstSpan(std::initializer_list<T> v) {
  return Span<const T>(v.begin(), v.size());
}
template<typename T>
constexpr Span<const T> makeConstSpan(Span<T> v) {
  return Span<const T>(v.data(), v.size());
}

}  // namespace ly
