#ifndef LLAMA_CC_READABLE_FILE_H_
#define LLAMA_CC_READABLE_FILE_H_

#include <stdio.h>
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include "common.h"
#include "status.h"
#include "util.h"

namespace llama {

// A wrapper of FILE in stdio.h
class ReadableFile {
 public:
  // Open a file for read.
  static StatusOr<ReadableFile> Open(const std::string &filename);

  virtual ~ReadableFile() = default;

  // This is a low level file read function, please use other high level
  // functions instead.
  // Read a buffer from file and store the number of bytes read into `pcbytes`.
  // On success, return `OkStatus()` and the `pcbytes` should equal to
  // `buffer.size()`. On failed, return the error status, buffer will be filled
  // with the bytes read before error occured. 0 <= `pcbytes` < `buffer.size()`
  virtual Status Read(util::Span<ByteType> buffer, int *pcbytes) = 0;
  Status Read(util::Span<ByteType> buffer);

  // read a variable from reader
  template<typename T>
  Status ReadValue(T *value) {
    RETURN_IF_ERROR(Read(util::MakeSpan(
        reinterpret_cast<ByteType *>(value),
        sizeof(T))));
    
    return OkStatus();
  }

  // read a string of `n` bytes from file
  Status ReadString(int n, std::string *s);

  // read all byte from file and save to to data
  Status ReadAll(std::vector<ByteType> *data);
};

}  // namespace llama


#endif  // LLAMA_CC_READABLE_FILE_H_
