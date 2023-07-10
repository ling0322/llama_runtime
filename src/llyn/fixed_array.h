#pragma once

#include "llyn/internal/base_array.h"

namespace ly {

template<typename T>
class FixedArray : public internal::BaseArray<T> {
 public:
  FixedArray() noexcept : BaseArray() {}
  FixedArray(int size) noexcept : 
      BaseArray(size ? new T[size] : nullptr, size) {}
  ~FixedArray() noexcept {
    delete[] _ptr;
    _ptr = nullptr;
    _size = 0;
  }

  // copy
  FixedArray(FixedArray<T> &) = delete;
  FixedArray<T> &operator=(FixedArray<T> &) = delete;

  // move
  FixedArray(FixedArray<T> &&array) noexcept {
    _ptr = array._ptr;
    _size = array._size;
    array._ptr = nullptr;
    array._size = 0;
  }
  FixedArray<T> &operator=(FixedArray<T> &&array) noexcept {
    if (_ptr) {
      delete[] _ptr;
    }

    _ptr = array._ptr;
    _size = array._size;
    array._ptr = nullptr;
    array._size = 0;

    return *this;
  }

  FixedArray<T> copy() const {
    FixedArray<T> l(_size);
    std::copy(begin(), end(), l.begin());

    return l;
  }
};

}  // namespace ly
