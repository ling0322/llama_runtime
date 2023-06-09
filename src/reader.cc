#include "reader.h"

#include "common.h"
#include "strings.h"
#include "log.h"
#include "util.h"

namespace llama {

std::vector<ByteType> readFile(const std::string &filename) {
  std::vector<ByteType> data;
  std::vector<ByteType> chunk(4096);

  auto fp = ReadableFile::open(filename);

  for (; ; ) {
    int cbytes = fp->read(util::makeSpan(chunk));
    if (cbytes) {
      data.insert(data.end(), chunk.begin(), chunk.begin() + cbytes);
    } else {
      break;
    }
  }

  return data;
}

// -- class BufferedReader -------------------------------------------------------------------------

BufferedReader::BufferedReader(int buffer_size)
    : _buffer(buffer_size), 
      _w(0),
      _r(0) {}

int BufferedReader::readFromBuffer(util::Span<ByteType> dest) {
  int n = std::min(static_cast<int>(dest.size()), _w - _r);

  ByteType *begin = _buffer.begin() + _r;
  std::copy(begin, begin + n, dest.begin());

  _r += n;
  return n;
}

int BufferedReader::readNextBuffer() {
  CHECK(_w - _r == 0);
  _r = 0;
  _w = 0;
  _w = read(util::makeSpan(_buffer));

  return _w;
}

void BufferedReader::readSpan(util::Span<ByteType> span) {
  util::Span<ByteType>::iterator it = span.begin();

  int bytesRead = readFromBuffer(span);
  
  while (bytesRead < span.size()) {
    int n = readNextBuffer();
    if (!n) {
      throw AbortedException("unexcpected end-of-file");
    }

    bytesRead += readFromBuffer(span.subspan(bytesRead));
  }
}

std::string BufferedReader::readString(int n) {
  CHECK(n > 0);

  std::vector<ByteType> buffer(n);
  readSpan(util::makeSpan(buffer));

  return std::string(buffer.begin(), buffer.end());
}

// -- class ReadableFile ---------------------------------------------------------------------------

ReadableFile::ReadableFile() : _fp(nullptr) {}
ReadableFile::~ReadableFile() {
  if (_fp) {
    fclose(_fp);
    _fp = nullptr;
  }
}

std::unique_ptr<ReadableFile> ReadableFile::open(const std::string &filename) {
  std::unique_ptr<ReadableFile> fp{new ReadableFile()};
  fp->_fp = fopen(filename.c_str(), "rb");
  if (fp->_fp == nullptr) {
    throw AbortedException(str::sprintf("failed to open file %s.", filename));
  }

  return fp;
}

int ReadableFile::read(util::Span<ByteType> buffer) {
  CHECK(buffer.size() != 0);
  int n = fread(buffer.data(), sizeof(ByteType), buffer.size(), _fp);

  if ((!n) && !feof(_fp)) {
    throw AbortedException("failed to read file.");
  }

  return n;
}

// -- class Scanner --------------------------------------------------------------------------------

Scanner::Scanner(Reader *reader) : _reader(reader), _buffer(BufferSize) {}

bool Scanner::scan() {
  _text.clear();

  for (; ; ) {
    if (_bufferSpan.empty()) {
      bool ok = readBuffer();
      if (!ok) {
        return !_text.empty();
      }
    }

    auto it = _bufferSpan.begin();
    for (; it < _bufferSpan.end() && *it != '\n'; ++it) {
    }
    if (it != _bufferSpan.end()) {
      _text.insert(_text.end(), _bufferSpan.begin(), it);
      _bufferSpan = _bufferSpan.subspan(it - _bufferSpan.begin() + 1);
      return true;
    } else {
      _text.insert(_text.end(), _bufferSpan.begin(), _bufferSpan.end());
      _bufferSpan = util::Span<ByteType>();
    }
  }

  // will not reach here.
  NOT_IMPL();
  return false;
}

const std::string &Scanner::getText() const {
  return _text;
}

bool Scanner::readBuffer() {
  int n = _reader->read(util::makeSpan(_buffer));
  if (!n) {
    return false;
  }

  _bufferSpan = util::makeSpan(_buffer.data(), n);
  return true;
}

}  // namespace llama
