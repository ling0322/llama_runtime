#pragma once

namespace ly {

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
    CHECK(_currentOffset <= BLOCK_SIZE);
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

}
