#pragma once

#include "llyn/internal/base_array.h"

namespace ly {

template<typename T>
class FixedArray : public internal::BaseArray<T> {
 public:
  FixedArray() noexcept : internal::BaseArray<T>() {}
  FixedArray(int size) noexcept : 
      internal::BaseArray<T>(size ? new T[size] : nullptr, size) {}
  ~FixedArray() noexcept {
    delete[] internal::BaseArray<T>::_ptr;
    internal::BaseArray<T>::_ptr = nullptr;
    internal::BaseArray<T>::_size = 0;
  }

  // copy
  FixedArray(FixedArray<T> &) = delete;
  FixedArray<T> &operator=(FixedArray<T> &) = delete;

  // move
  FixedArray(FixedArray<T> &&array) noexcept {
    internal::BaseArray<T>::_ptr = array._ptr;
    internal::BaseArray<T>::_size = array._size;
    array._ptr = nullptr;
    array._size = 0;
  }
  FixedArray<T> &operator=(FixedArray<T> &&array) noexcept {
    if (internal::BaseArray<T>::_ptr) {
      delete[] internal::BaseArray<T>::_ptr;
    }

    internal::BaseArray<T>::_ptr = array._ptr;
    internal::BaseArray<T>::_size = array._size;
    array._ptr = nullptr;
    array._size = 0;

    return *this;
  }

  FixedArray<T> copy() const {
    FixedArray<T> l(internal::BaseArray<T>::_size);
    std::copy(internal::BaseArray<T>::begin(), internal::BaseArray<T>::end(), l.begin());

    return l;
  }
};

}  // namespace ly
