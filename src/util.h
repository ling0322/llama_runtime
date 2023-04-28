#ifndef LLAMA_RUNTIME_UTIL_H_
#define LLAMA_RUNTIME_UTIL_H_

#include <functional>
#include <string>
#include <vector>

#include "common.h"
#include "log.h"
#include "status.h"

namespace llama {
namespace util {

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

 protected:
  pointer ptr_;
  size_type size_;
};

// ----------------------------------------------------------------------------
// FixedArray
// ----------------------------------------------------------------------------

template<typename T>
class FixedArray : public BaseArray<T> {
 public:
  FixedArray() noexcept : BaseArray() {}
  FixedArray(int size) noexcept : 
      BaseArray(size ? new T[size] : nullptr, size) {}
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

  FixedArray<T> Copy() const {
    FixedArray<T> l(size_);
    std::copy(begin(), end(), l.begin());

    return l;
  }
};

// ----------------------------------------------------------------------------
// Span
// ----------------------------------------------------------------------------

template<typename T>
class Span : public BaseArray<T> {
 public:
  Span() noexcept : ptr_(nullptr), len_(0) {}
  Span(T *ptr, size_type size) : BaseArray(ptr, size) {}

  Span<T> subspan(size_type pos = 0, size_type len = npos) const {
    ASSERT(pos <= size());
    len = std::min(size() - pos, len);
    return Span<T>(data() + pos, len);
  }
};

template<typename T>
constexpr Span<T> MakeSpan(
    T *ptr,
    typename Span<T>::size_type size) {
  return Span<T>(ptr, size);
}
template<typename T>
constexpr Span<const T> MakeConstSpan(
    const T *ptr,
    typename Span<T>::size_type size) {
  return Span<const T>(ptr, size);
}
template<typename T>
constexpr Span<T> MakeSpan(std::vector<T> &v) {
  return Span<T>(v.data(), v.size());
}
template<typename T>
constexpr Span<const T> MakeConstSpan(const std::vector<T> &v) {
  return Span<const T>(v.data(), v.size());
}

template<typename T>
constexpr Span<T> MakeSpan(const FixedArray<T> &v) {
  return Span<T>(v.data(), v.size());
}
template<typename T>
constexpr Span<const T> MakeConstSpan(const FixedArray<T> &v) {
  return Span<const T>(v.data(), v.size());
}
template<typename T>
constexpr Span<const T> MakeConstSpan(std::initializer_list<T> v) {
  return Span<const T>(v.begin(), v.size());
}


// ---------------------------------------------------------------------------+
// NonCopyable                                                                |
// ---------------------------------------------------------------------------+

class NonCopyable {
 public: 
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;

 protected:
  NonCopyable() = default;
  ~NonCopyable() = default;
};

//
// class AutoCPtr
//

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
  T **get_pp() { CHECK(deleter_); return &ptr_; }

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

// ---------------------------------------------------------------------------+
// class Path                                                                 |
// ---------------------------------------------------------------------------+

// provide base functions for path. For example, path::Join, dirname, basename
// and convert to wstring, ...
class Path {
 public:
  static Path CurrentModulePath();
  static Path CurrentExecutablePath();

  Path(const std::string &path);
  Path(std::string &&path);

  bool operator==(const Path &r) const;
  bool operator==(const std::string &r) const;

  Path dirname() const;
  Path basename() const;

  Path operator/(const Path &path) const;
  Path operator/(const std::string &path) const;

  std::string string() const;
  Status AsWString(std::wstring *ws) const;

 private:
  std::string path_;
};

// ---------------------------------------------------------------------------+
// class Pool                                                                 |
// ---------------------------------------------------------------------------+

// Pool is a class to optimize allocating large number of a class T.
template<typename T, int BLOCK_SIZE = 4096>
class Pool {
 public:
  static constexpr int16_t kUnmarked = 0;
  static constexpr int16_t kMarked = 1;

  Pool();
  ~Pool();

  // Allocate a class T with constructor parameter args 
  T *Alloc();

  // Allocate a class T with constructor parameter args 
  void Free(T *pointer);
  void Free(const T *pointer);

  // Clear all allocated class T
  void Clear();

  // Start gabbage collection. Root set for GC is in root_nodes
  void GC(const std::vector<T *> root);

  // Free and allocated nodes in this pool
  int num_free() const;
  int num_allocated() const;

 protected:
  std::vector<T *> blocks_;
  std::vector<T *> free_;
  int current_block_;
  int current_offset_;
};


template<typename T, int BLOCK_SIZE>
Pool<T, BLOCK_SIZE>::Pool() : current_block_(0), current_offset_(0) {
  T *block = reinterpret_cast<T *>(::operator new(sizeof(T) * BLOCK_SIZE));
  blocks_.push_back(block);
}

template<typename T, int BLOCK_SIZE>
Pool<T, BLOCK_SIZE>::~Pool() {
  Clear();
  for (T *block : blocks_) {
    ::operator delete(block);
  }
  current_block_ = 0;
  current_offset_ = 0;
}

template<typename T, int BLOCK_SIZE>
T *Pool<T, BLOCK_SIZE>::Alloc() {
  T *memory;
  if (free_.empty()) {
    ASSERT(current_offset_ <= BLOCK_SIZE);
    if (current_offset_ == BLOCK_SIZE) {
      if (current_block_ == blocks_.size() - 1) {
        T *block = reinterpret_cast<T *>(
            ::operator new(sizeof(T) * BLOCK_SIZE));
        blocks_.push_back(block);
      }
      ++current_block_;
      current_offset_ = 0;
    }
    memory = blocks_[current_block_] + current_offset_;
    ++current_offset_;
  } else {
    memory = free_.back();
    free_.pop_back();
  }

  return memory;
}

template<typename T, int BLOCK_SIZE>
void Pool<T, BLOCK_SIZE>::Free(T *pointer) {
  free_.push_back(pointer);
}

template<typename T, int BLOCK_SIZE>
void Pool<T, BLOCK_SIZE>::Clear() {
  current_block_ = 0;
  current_offset_ = 0;
  free_.clear();
}

template<typename T, int BLOCK_SIZE>
int Pool<T, BLOCK_SIZE>::num_free() const {
  return free_.size() +
          (blocks_.size() - current_block_) * BLOCK_SIZE -
          current_offset_;
}

template<typename T, int BLOCK_SIZE>
int Pool<T, BLOCK_SIZE>::num_allocated() const {
  return current_block_ * BLOCK_SIZE + current_offset_ - free_.size();
}

}  // namespace util
}  // namespace llama

#endif  // LLAMA_RUNTIME_UTIL_H_
