#include "reader.h"

#include "common.h"
#include "status.h"
#include "log.h"
#include "util.h"

namespace llama {

// ----------------------------------------------------------------------------
// class BufferedReader
// ----------------------------------------------------------------------------

Status ReadFile(const std::string &filename, std::vector<ByteType> *data) {
  std::vector<ByteType> chunk(4096);
  auto fp = ReadableFile::Open(filename);
  RETURN_IF_ERROR(fp);

  data->clear();
  for (; ; ) {
    int cbytes = 0;
    Status status = fp->Read(util::MakeSpan(chunk), &cbytes);
    if (status.ok()) {
      data->insert(data->end(), chunk.begin(), chunk.end());
    } else {
      if (IsOutOfRange(status)) {
        // reading finished, put remaining bytes to `data`
        data->insert(data->end(), chunk.begin(), chunk.begin() + cbytes);
        break;
      } else {
        // error occured
        RETURN_IF_ERROR(status);
      }
    }
  }

  return OkStatus();
}

// ----------------------------------------------------------------------------
// class BufferedReader
// ----------------------------------------------------------------------------

BufferedReader::BufferedReader(int buffer_size)
    : buffer_(buffer_size), 
      w_(0),
      r_(0) {}

int BufferedReader::ReadFromBuffer(util::Span<ByteType> dest) {
  int n = std::min(static_cast<int>(dest.size()), w_ - r_);

  ByteType *begin = buffer_.begin() + r_;
  std::copy(begin, begin + n, dest.begin());

  r_ += n;
  return n;
}

Status BufferedReader::ReadNextBuffer() {
  CHECK(w_ - r_ == 0);
  r_ = 0;
  w_ = 0;
  RETURN_IF_ERROR(Read(util::MakeSpan(buffer_), &w_));

  return OkStatus();
}

Status BufferedReader::ReadSpan(util::Span<ByteType> buffer) {
  util::Span<ByteType>::iterator it = buffer.begin();

  int n = ReadFromBuffer(buffer);
  
  while (n < buffer.size()) {
    RETURN_IF_ERROR(ReadNextBuffer());
    n += ReadFromBuffer(buffer.subspan(n));
  }

  return OkStatus();
}

Status BufferedReader::ReadString(int n, std::string *s) {
  CHECK(n > 0);

  std::vector<ByteType> buffer(n);
  RETURN_IF_ERROR(ReadSpan(util::MakeSpan(buffer)));

  *s = std::string(buffer.begin(), buffer.end());
  return OkStatus();
}

Status BufferedReader::ReadLineFromBuffer(std::string *s) {
  while (r_ < w_) {
    ByteType b = buffer_[r_];
    s->push_back(b);
    ++r_;

    if (b == '\n') {
      return OkStatus();
    }
  }

  return OutOfRangeError();
}

Status BufferedReader::ReadLine(std::string *s) {
  s->clear();

  Status status = ReadLineFromBuffer(s);
  if (status.ok()) {
    return OkStatus();
  }

  for (; ; ) {
    // read next buffer
    status = ReadNextBuffer();
    if (IsOutOfRange(status)) {
      break;
    } else if (!status.ok()) {
      return status;
    }

    status = ReadLineFromBuffer(s);
    if (status.ok()) {
      return OkStatus();
    }
  }

  // EOF reached
  if (!s->empty()) {
    return OkStatus();
  } else {
    return OutOfRangeError();
  }
}

// ----------------------------------------------------------------------------
// class ReadableFile
// ----------------------------------------------------------------------------

ReadableFile::ReadableFile() : fp_(nullptr) {}
ReadableFile::~ReadableFile() {
  if (fp_) {
    fclose(fp_);
    fp_ = nullptr;
  }
}

StatusOr<ReadableFile> ReadableFile::Open(const std::string &filename) {
  std::unique_ptr<ReadableFile> fp{new ReadableFile()};
  fp->fp_ = fopen(filename.c_str(), "rb");
  if (fp->fp_ == nullptr) {
    RETURN_ABORTED() << "failed to open file " << filename;
  }

  return fp;
}

Status ReadableFile::Read(util::Span<ByteType> buffer, int *pcbytes) {
  CHECK(buffer.size() != 0);
  int n = fread(buffer.data(), sizeof(ByteType), buffer.size(), fp_);
  *pcbytes = n;

  if (!n) {
    if (feof(fp_)) {
      return OutOfRangeError();
    } else {
      return AbortedError("failed to read");
    }
  }

  return OkStatus();
}

}  // namespace llama
