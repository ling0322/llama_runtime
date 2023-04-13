#pragma once

#include <functional>
#include <string>

#include "common.h"
#include "log.h"

namespace llama {

// ----------------------------------------------------------------------------
// BaseArray
// ----------------------------------------------------------------------------

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
      ptr_(ptr),
      size_(size) {}

  pointer data() const noexcept { return ptr_; }
  size_type size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  reference operator[](size_type i) const noexcept {
    return *(ptr_ + i);
  }
  reference at(size_type i) const {
    ASSERT(i < size_ && i >= 0);
    return *(ptr_ + i);
  }
  reference front() const noexcept { ASSERT(!empty()); return *ptr_; }
  reference back() const noexcept { 
    ASSERT(!empty());
    return *(ptr_ + size_ - 1);
  }
  iterator begin() const noexcept { return ptr_; }
  iterator end() const noexcept { return ptr_ + size_; }
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

 private:
  pointer ptr_;
  size_type size_;
};

// ----------------------------------------------------------------------------
// FixedArray
// ----------------------------------------------------------------------------

template<typename T>
class FixedArray : public BaseArray<T> {
 public:
  FixedArray() noexcept : ptr_(nullptr), len_(0) {}
  FixedArray(int size) noexcept : 
      ptr_(size ? new T[size] : nullptr),
      size_(size) {}
  ~FixedArray() noexcept {
    delete[] ptr_;
    ptr_ = nullptr;
    size_ = 0;
  }

  // copy
  FixedArray(FixedArray<T> &) = delete;
  FixedArray<T> &operator=(FixedArray<T> &) = delete;

  // move
  FixedArray(FixedArray<T> &&array) noexcept {
    ptr_ = array.ptr_;
    size_ = array.size_;

    array.ptr_ = nullptr;
    array.size_ = 0;
  }
  FixedArray<T> &operator=(FixedArray<T> &&array) noexcept {
    if (ptr_) {
      delete[] ptr_;
    }

    ptr_ = array.ptr_;
    size_ = array.size_;

    array.ptr_ = nullptr;
    array.size_ = 0;

    return *this;
  }
};

// ----------------------------------------------------------------------------
// NonCopyable
// ----------------------------------------------------------------------------

class NonCopyable {
 public: 
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;

 protected:
  NonCopyable() = default;
  ~NonCopyable() = default;
};

template <typename T>
constexpr typename std::underlying_type<T>::type to_underlying(T v) {
  return static_cast<typename std::underlying_type<T>::type>(v);
}


// ----------------------------------------------------------------------------
// AutoCPtr
// ----------------------------------------------------------------------------


// Stores the C pointer and it's destroy function
template<typename T>
class AutoCPtr {
 public:
  AutoCPtr();
  AutoCPtr(T *ptr, std::function<void(T *)> deleter);
  AutoCPtr(AutoCPtr<T> &&auto_cptr) noexcept;
  ~AutoCPtr();

  AutoCPtr<T> &operator=(AutoCPtr<T> &&auto_cptr);

  // return the pointer and release the ownership
  T *Release();

  // get the pointer
  T *get() { return ptr_; }
  const T *get() const { return ptr_; }

  // get the pointer to this pointer.
  T **get_pp() { LL_CHECK(deleter_); return &ptr_; }

 private:
  T *ptr_;
  std::function<void(T *)> deleter_;

  DISALLOW_COPY_AND_ASSIGN(AutoCPtr)
};

template<typename T>
inline AutoCPtr<T>::AutoCPtr(): ptr_(nullptr), deleter_(nullptr) {}
template<typename T>
inline AutoCPtr<T>::AutoCPtr(T *ptr, std::function<void(T *)> deleter): 
    ptr_(ptr),
    deleter_(deleter) {}
template<typename T>
inline AutoCPtr<T>::AutoCPtr(AutoCPtr<T> &&auto_cptr) noexcept :
    ptr_(auto_cptr.ptr_),
    deleter_(auto_cptr.deleter_) {
  auto_cptr.ptr_ = nullptr;
}
template<typename T>
inline AutoCPtr<T>::~AutoCPtr() {
  if (ptr_) {
    deleter_(ptr_);
    ptr_ = nullptr;
  }
}
template<typename T>
AutoCPtr<T> &AutoCPtr<T>::operator=(AutoCPtr<T> &&auto_cptr) {
  if (ptr_) {
    deleter_(ptr_);
    ptr_ = nullptr;
  }

  ptr_ = auto_cptr.ptr_;
  deleter_ = auto_cptr.deleter_;

  auto_cptr.ptr_ = nullptr;
  return *this;
}
template<typename T>
inline T *AutoCPtr<T>::Release() {
  T *ptr = ptr_;
  ptr_ = nullptr;

  return ptr;
}

}  // namespace llama
