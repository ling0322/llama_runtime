#ifndef LLAMA_CC_POOL_H_
#define LLAMA_CC_POOL_H_

#include <stdio.h>
#include <vector>
#include <unordered_set>
#include <memory>
#include <type_traits>
#include "common.h"

namespace llama {

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

}  // namespace llama

#endif  // LLAMA_CC_POOL_H_
