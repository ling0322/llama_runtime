#pragma once

#include <functional>
#include "llyn/noncopyable.h"

namespace ly {

// Stores the C pointer and it's destroy function
template<typename T>
class c_ptr : private NonCopyable {
 public:
  c_ptr();
  c_ptr(T *ptr, std::function<void(T *)> deleter);
  c_ptr(c_ptr<T> &&auto_cptr) noexcept;
  ~c_ptr();

  c_ptr<T> &operator=(c_ptr<T> &&auto_cptr);

  // return the pointer and release the ownership
  T *Release();

  // get the pointer
  T *get() { return _ptr; }
  const T *get() const { return _ptr; }

  // get the pointer to this pointer.
  T **get_pp() { CHECK(_deleter); return &_ptr; }

 private:
  T *_ptr;
  std::function<void(T *)> _deleter;
};


template<typename T>
inline c_ptr<T>::c_ptr(): _ptr(nullptr), _deleter(nullptr) {}
template<typename T>
inline c_ptr<T>::c_ptr(T *ptr, std::function<void(T *)> deleter): 
    _ptr(ptr),
    _deleter(deleter) {}
template<typename T>
inline c_ptr<T>::c_ptr(c_ptr<T> &&auto_cptr) noexcept :
    _ptr(auto_cptr._ptr),
    _deleter(auto_cptr._deleter) {
  auto_cptr._ptr = nullptr;
}
template<typename T>
inline c_ptr<T>::~c_ptr() {
  if (_ptr) {
    _deleter(_ptr);
    _ptr = nullptr;
  }
}
template<typename T>
c_ptr<T> &c_ptr<T>::operator=(c_ptr<T> &&auto_cptr) {
  if (_ptr) {
    _deleter(_ptr);
    _ptr = nullptr;
  }

  _ptr = auto_cptr._ptr;
  _deleter = auto_cptr._deleter;

  auto_cptr._ptr = nullptr;
  return *this;
}
template<typename T>
inline T *c_ptr<T>::Release() {
  T *ptr = _ptr;
  _ptr = nullptr;

  return ptr;
}

}
