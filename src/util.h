#pragma once

#include <functional>
#include <string>

#include "common.h"
#include "log.h"

namespace llama {

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
