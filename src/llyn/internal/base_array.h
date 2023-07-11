#pragma once

#include <assert.h>
#include <functional>
#include <string>
#include <vector>

namespace ly {
namespace internal {

template<typename T>
class BaseArray {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using iterator = pointer;
  using const_iterator = const_pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  static const size_type npos = ~(size_type(0));

  BaseArray() : BaseArray(nullptr, 0) {}
  BaseArray(T *ptr, size_type size) noexcept : 
      _ptr(ptr),
      _size(size) {}

  pointer data() const noexcept { return _ptr; }
  size_type size() const noexcept { return _size; }
  bool empty() const noexcept { return _size == 0; }
  reference operator[](size_type i) const noexcept {
    return *(_ptr + i);
  }
  reference at(size_type i) const {
    assert(i < _size && i >= 0);
    return *(_ptr + i);
  }
  reference front() const noexcept { ASSERT(!empty()); return *_ptr; }
  reference back() const noexcept { 
    assert(!empty());
    return *(_ptr + _size - 1);
  }
  iterator begin() const noexcept { return _ptr; }
  iterator end() const noexcept { return _ptr + _size; }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator cend() const noexcept { return end(); }
  reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }
  reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }
  const_reverse_iterator crbegin() const noexcept { return rbegin(); }
  const_reverse_iterator crend() const noexcept { return rend(); }

 protected:
  pointer _ptr;
  size_type _size;
};

}  // namespace internal
}  // namespace ly
