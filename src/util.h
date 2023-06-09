#pragma once

#include <functional>
#include <string>
#include <vector>

#include "common.h"
#include "log.h"

namespace llama {
namespace util {

// -- OS dependent functions ---------------------------------------------------

bool isAvx512Available();
bool isAvx2Available();

// ---------------------------------------------------------------------------+
// BaseArray                                                                  |
// ---------------------------------------------------------------------------+

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
    ASSERT(i < _size && i >= 0);
    return *(_ptr + i);
  }
  reference front() const noexcept { ASSERT(!empty()); return *_ptr; }
  reference back() const noexcept { 
    ASSERT(!empty());
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

// -- FixedArray ----------

template<typename T>
class FixedArray : public BaseArray<T> {
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

// -- class Span ----------

template<typename T>
class Span : public BaseArray<T> {
 public:
  Span() noexcept : BaseArray() {}
  Span(T *ptr, size_type size) : BaseArray(ptr, size) {}

  // automatic convert initializer_list to Span<const T>.
  // NOTE: initializer_list should outlives span when using this constructor.
  // Examples:
  //   Span<const int> v = {1, 2, 3};  // WRONG: lifetime of initializer_list is shorter than v;
  template <typename U = T,
            typename = std::enable_if<std::is_const<T>::value, U>::type>
  Span(std::initializer_list<value_type> v LR_LIFETIME_BOUND) noexcept
      : Span(v.begin(), v.size()) {}

  Span<T> subspan(size_type pos = 0, size_type len = npos) const {
    ASSERT(pos <= size());
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
constexpr Span<const T> makeConstSpan(util::Span<T> v) {
  return Span<const T>(v.data(), v.size());
}

// -- NonCopyable ----------

class NonCopyable {
 public: 
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;

 protected:
  NonCopyable() = default;
  ~NonCopyable() = default;
};

// -- class AutoCPtr ----------

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
  T *get() { return _ptr; }
  const T *get() const { return _ptr; }

  // get the pointer to this pointer.
  T **get_pp() { CHECK(deleter_); return &_ptr; }

 private:
  T *_ptr;
  std::function<void(T *)> _deleter;

  DISALLOW_COPY_AND_ASSIGN(AutoCPtr)
};

template<typename T>
inline AutoCPtr<T>::AutoCPtr(): _ptr(nullptr), _deleter(nullptr) {}
template<typename T>
inline AutoCPtr<T>::AutoCPtr(T *ptr, std::function<void(T *)> deleter): 
    _ptr(ptr),
    _deleter(deleter) {}
template<typename T>
inline AutoCPtr<T>::AutoCPtr(AutoCPtr<T> &&auto_cptr) noexcept :
    _ptr(auto_cptr._ptr),
    _deleter(auto_cptr._deleter) {
  auto_cptr._ptr = nullptr;
}
template<typename T>
inline AutoCPtr<T>::~AutoCPtr() {
  if (_ptr) {
    _deleter(_ptr);
    _ptr = nullptr;
  }
}
template<typename T>
AutoCPtr<T> &AutoCPtr<T>::operator=(AutoCPtr<T> &&auto_cptr) {
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
inline T *AutoCPtr<T>::Release() {
  T *ptr = _ptr;
  _ptr = nullptr;

  return ptr;
}

// ---------------------------------------------------------------------------+
// class Path                                                                 |
// ---------------------------------------------------------------------------+

// provide base functions for path. For example, path::Join, dirname, basename
// and convert to wstring, ...
class Path {
 public:
  static Path currentModulePath();
  static Path currentExecutablePath();

  Path() = default;
  Path(const std::string &path);
  Path(std::string &&path);

  bool operator==(const Path &r) const;
  bool operator==(const std::string &r) const;

  Path dirname() const;
  Path basename() const;
  bool isabs() const;

  Path operator/(const Path &path) const;
  Path operator/(const std::string &path) const;

  std::string string() const;
  std::wstring wstring() const;

 private:
  std::string _path;

  // normalize path string.
  static std::string normPath(const std::string &path);
};

// -- class Pool ----------

// Pool is a class to optimize allocating large number of a class T.
template<typename T, int BLOCK_SIZE = 4096>
class Pool {
 public:
  static constexpr int16_t kUnmarked = 0;
  static constexpr int16_t kMarked = 1;

  Pool();
  ~Pool();

  // Allocate a class T with constructor parameter args 
  T *alloc();

  // Allocate a class T with constructor parameter args 
  void free(T *pointer);
  void free(const T *pointer);

  // Clear all allocated class T
  void clear();

  // Start gabbage collection. Root set for GC is in root_nodes
  void gc(const std::vector<T *> root);

  // Free and allocated nodes in this pool
  int getNumFree() const;
  int getNumAllocated() const;

 protected:
  std::vector<T *> _blocks;
  std::vector<T *> _free;
  int _currentBlock;
  int _currentOffset;
};


template<typename T, int BLOCK_SIZE>
Pool<T, BLOCK_SIZE>::Pool() : _currentBlock(0), _currentOffset(0) {
  T *block = reinterpret_cast<T *>(::operator new(sizeof(T) * BLOCK_SIZE));
  _blocks.push_back(block);
}

template<typename T, int BLOCK_SIZE>
Pool<T, BLOCK_SIZE>::~Pool() {
  clear();
  for (T *block : _blocks) {
    ::operator delete(block);
  }
  _currentBlock = 0;
  _currentOffset = 0;
}

template<typename T, int BLOCK_SIZE>
T *Pool<T, BLOCK_SIZE>::alloc() {
  T *memory;
  if (_free.empty()) {
    ASSERT(_currentOffset <= BLOCK_SIZE);
    if (_currentOffset == BLOCK_SIZE) {
      if (_currentBlock == _blocks.size() - 1) {
        T *block = reinterpret_cast<T *>(::operator new(sizeof(T) * BLOCK_SIZE));
        _blocks.push_back(block);
      }
      ++_currentBlock;
      _currentOffset = 0;
    }
    memory = _blocks[_currentBlock] + _currentOffset;
    ++_currentOffset;
  } else {
    memory = _free.back();
    _free.pop_back();
  }

  return memory;
}

template<typename T, int BLOCK_SIZE>
void Pool<T, BLOCK_SIZE>::free(T *pointer) {
  _free.push_back(pointer);
}

template<typename T, int BLOCK_SIZE>
void Pool<T, BLOCK_SIZE>::clear() {
  _currentBlock = 0;
  _currentOffset = 0;
  _free.clear();
}

template<typename T, int BLOCK_SIZE>
int Pool<T, BLOCK_SIZE>::getNumFree() const {
  return _free.size() + (blocks_.size() - _currentBlock) * BLOCK_SIZE - _currentOffset;
}

template<typename T, int BLOCK_SIZE>
int Pool<T, BLOCK_SIZE>::getNumAllocated() const {
  return _currentBlock * BLOCK_SIZE + _currentOffset - _free.size();
}


}  // namespace util
}  // namespace llama
