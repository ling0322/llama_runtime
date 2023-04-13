#include "readable_file.h"

#include "common.h"
#include "status.h"
#include "log.h"
#include "span.h"

namespace llama {

class LocalFile : public ReadableFile {
 public:
  LocalFile();
  ~LocalFile();

  Status Open(PCStrType filename);
  Status Read(Span<ByteType> buffer, int *pcbytes) override;

 private:
  FILE *fd_;
};

// ----------------------------------------------------------------------------
// class ReadableFile
// ----------------------------------------------------------------------------

StatusOr<ReadableFile> ReadableFile::Open(const std::string& filename) {
  std::unique_ptr<LocalFile> fp = std::make_unique<LocalFile>();
  std::string name = std::string(filename);
  RETURN_IF_ERROR(fp->Open(name.c_str()));

  return fp;
}

Status ReadableFile::Read(Span<ByteType> buffer) {
  int cbytes = 0;
  return Read(buffer, &cbytes);
}

Status ReadableFile::ReadString(int n, std::string *s) {
  std::vector<ByteType> buffer(n);
  RETURN_IF_ERROR(Read(MakeSpan(buffer)));

  *s = std::string(buffer.begin(), buffer.end());
  return OkStatus();
}

Status ReadableFile::ReadAll(std::vector<ByteType> *data) {
  std::vector<ByteType> chunk(4096);

  data->clear();
  for (; ; ) {
    int cbytes = 0;
    Status status = Read(MakeSpan(chunk), &cbytes);
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
// class LocalFile
// ----------------------------------------------------------------------------

LocalFile::LocalFile() : fd_(nullptr) {}
LocalFile::~LocalFile() {
  if (fd_) {
    fclose(fd_);
    fd_ = nullptr;
  }
}

Status LocalFile::Open(PCStrType filename) {
  fd_ = fopen(filename, "rb");
  if (fd_ == nullptr) {
    RETURN_ABORTED() << "failed to open file " << filename;
  }

  return OkStatus();
}

Status LocalFile::Read(Span<ByteType> buffer, int *pcbytes) {
  LL_CHECK(buffer.size() != 0);
  int n = fread(buffer.data(), sizeof(ByteType), buffer.size(), fd_);
  if (n < buffer.size()) {
    if (feof(fd_)) {
      *pcbytes = n;
      return OutOfRangeError();
    } else {
      return AbortedError("failed to read");
    }
  } else {
    *pcbytes = n;
  }

  return OkStatus();
}

}  // namespace llama
