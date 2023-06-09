#ifndef LLAMA_CC_READABLE_FILE_H_
#define LLAMA_CC_READABLE_FILE_H_

#include <stdio.h>
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include "common.h"
#include "util.h"

namespace llama {

// read all bytes from `filename`.
std::vector<ByteType> readFile(const std::string &filename);

// -- class Reader ---------------------------------------------------------------------------------

// base class for all readers
class Reader {
 public:
  virtual ~Reader() = default;
  
  // Read a buffer from file and return number of bytes read. On EOF reached, returns 0. Throw
  // exceptio if other errors occured.
  // This is a low level file read function, please use other high level functions instead.
  virtual int read(util::Span<ByteType> buffer) = 0;
};

// -- class BufferedReader -------------------------------------------------------------------------

// implements buffered read.
class BufferedReader : public Reader {
 public:
  static constexpr int kDefaultBufferSize = 4096;

  // create the BufferedReader instance by Reader and the buffer size.
  BufferedReader(int buffer_size = kDefaultBufferSize);
  
  // read `buffer.size()` bytes and fill the `buffer`. Throw if EOF reached or other errors occured.
  void readSpan(util::Span<ByteType> buffer);

  // read a variable from reader
  template<typename T>
  T readValue();

  // read a string of `n` bytes from file
  std::string readString(int n);

 private:
  util::FixedArray<ByteType> _buffer;

  // write and read position in buffer
  int _w;
  int _r;

  // read at most `dest.size()` bytes from buffer and return the number of bytes read. The number
  // will less than `dest.size()` once no enough bytes in buffer.
  int readFromBuffer(util::Span<ByteType> dest);

  // read next buffer from Reader. Return the number of bytes read.
  int readNextBuffer();
};

template<typename T>
T BufferedReader::readValue() {
  T value;
  readSpan(util::makeSpan(reinterpret_cast<ByteType *>(&value), sizeof(T)));
  
  return value;
}

// -- class Scanner --------------------------------------------------------------------------------

// interface to read file line by line.
// Example:
//   Scanner scanner(fp);
//   while (scanner.scan()) {
//     const std::string &s = scanner.getText(); 
//   }
class Scanner {
 public:
  static constexpr int BufferSize = 4096;

  Scanner(Reader *reader);

  // returns false once EOF reached.
  bool scan();
  const std::string &getText() const;

 private:
  Reader *_reader;
  std::string _text;

  util::FixedArray<ByteType> _buffer;
  util::Span<ByteType> _bufferSpan;

  bool readBuffer();
};


// -- class ReadableFile ---------------------------------------------------------------------------

// reader for local file.
class ReadableFile : public BufferedReader,
                     private util::NonCopyable {
 public:
  static std::unique_ptr<ReadableFile> open(const std::string &filename);

  ~ReadableFile();

  // implements interface Reader
  int read(util::Span<ByteType> buffer) override;

 private:
  FILE *_fp;

  ReadableFile();
};

}  // namespace llama

#endif  // LLAMA_CC_READABLE_FILE_H_

