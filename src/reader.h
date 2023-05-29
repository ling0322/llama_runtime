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

// ---------------------------------------------------------------------------+
// function ReadFile                                                          |
// ---------------------------------------------------------------------------+

Status ReadFile(const std::string &filename, std::vector<ByteType> *data);

// ---------------------------------------------------------------------------+
// class Reader                                                               |
// ---------------------------------------------------------------------------+

// base class for all readers
class Reader {
 public:
  virtual ~Reader() = default;

  // This is a low level file read function, please use other high level
  // functions instead.
  // Read a buffer from file and store the number of bytes read into `pcbytes`.
  // pcbytes may less than `buffer.size()`
  virtual Status Read(util::Span<ByteType> buffer, int *pcbytes) = 0;
};

// ----------------------------------------------------------------------------
// class BufferedReader
// ----------------------------------------------------------------------------

// implements buffered read.
class BufferedReader : public Reader {
 public:
  static constexpr int kDefaultBufferSize = 4096;

  // create the BufferedReader instance by Reader and the buffer size.
  BufferedReader(int buffer_size = kDefaultBufferSize);
  
  // read `buffer.size()` bytes and fill the `buffer`. Return a failed state if
  // EOF reached or other errors occured.
  Status ReadSpan(util::Span<ByteType> buffer);

  // read a variable from reader
  template<typename T>
  Status ReadValue(T *value);

  // read a string of `n` bytes from file
  Status ReadString(int n, std::string *s);

  // read a line from file. Keeps the line delim at the back of `s`
  // Example:
  //   while (IsOK(fp->ReadLine(&line))) { ... }
  Status ReadLine(std::string *s);

 private:
  util::FixedArray<ByteType> buffer_;

  // write and read position in buffer
  int w_;
  int r_;

  // read at most `dest.size()` bytes from buffer and return the number of
  // bytes read. The number will less than `dest.size()` once no enough bytes
  // in buffer.
  int ReadFromBuffer(util::Span<ByteType> dest);

  // read a line from buffer. Return OutOfRangeError() if no enough bytes in
  // buffer.
  Status ReadLineFromBuffer(std::string *s);

  // read next buffer from Reader.
  Status ReadNextBuffer();
};

template<typename T>
Status BufferedReader::ReadValue(T *value) {
  RETURN_IF_ERROR(ReadSpan(util::MakeSpan(
      reinterpret_cast<ByteType *>(value),
      sizeof(T))));
  
  return OkStatus();
}

// -- class Scanner ------------------------------------------------------------

// interface to read file line by line.
// Example:
//   Scanner scanner(fp);
//   while (scanner.Scan()) {
//     const std::string &s = scanner.text(); 
//   }
//   RETURN_IF_ERROR(scanner.status());
class Scanner {
 public:
  Scanner(BufferedReader *reader);

  bool Scan();
  const std::string &text() const;
  Status status() const;

 private:
  BufferedReader *reader_;
  std::string text_;
  Status status_;
};


// ----------------------------------------------------------------------------
// class ReadableFile
// ----------------------------------------------------------------------------

// reader for local file.
class ReadableFile : public BufferedReader,
                     private util::NonCopyable {
 public:
  static expected_ptr<ReadableFile> Open(const std::string &filename);

  ~ReadableFile();

  // implements interface Reader
  Status Read(util::Span<ByteType> buffer, int *pcbytes) override;

 private:
  FILE *fp_;

  ReadableFile();
};

}  // namespace llama

#endif  // LLAMA_CC_READABLE_FILE_H_

