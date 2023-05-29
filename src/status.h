#ifndef LLAMA_RUNTIME_STATUS_H_
#define LLAMA_RUNTIME_STATUS_H_

#include <exception>
#include <string>
#include <sstream>

#include "common.h"
#include "log.h"

#define RETURN_IF_ERROR(expr)                 \
  if (llama::StatusWrapper sw = (expr)) { \
  } else                                      \
    return std::move(sw)

#define MAKE_STATUS(code) \
  llama::StatusBuilder(llama::StatusCode::code)
#define RETURN_STATUS(code) return MAKE_STATUS(code)
#define RETURN_OUT_OF_RANGE() RETURN_STATUS(kOutOfRange)
#define RETURN_ABORTED() RETURN_STATUS(kAborted)

namespace llama {

enum class StatusCode : int {
  kOK = 0,
  kAborted = 1,
  kOutOfRange = 2,
};

// Error code with message
class Status {
 public:
  static Status OK();

  explicit Status(StatusCode code);
  Status(StatusCode code, const std::string &what);
  Status(Status &&status);
  ~Status();

  Status &operator=(Status &&status);

  // copy constructor is disabled, use Copy() instead
  Status operator=(const Status &status) = delete;
  Status(Status& status) = delete;

  Status Copy() const;

  bool ok() const { return rep_ == 0; }
  std::string what() const;
  StatusCode code() const;

  // get internal representation of this status. For testing ONLY.
  uintptr_t rep() const { return rep_; }

 private:
  struct Rep;

  static constexpr uintptr_t kLowBit = 1;

  // return true if the error representation is on heap.
  bool IsHeapAllocated() const { return (rep_ & kLowBit) == 1; };

  Rep *GetRep() const;
  void SetRep(Rep *rep);
  StatusCode GetCode() const;
  std::string DefaultMessage(StatusCode code) const;

  // when rep & kLowBit == 0, it was a inline representation and the error code
  // was rep_ >> 1
  // when rep & kLowBit == 1, its representation (Rep *) was in address
  // (rep_ - 1)
  uintptr_t rep_;
};

class StatusBuilder {
 public:
  StatusBuilder(StatusCode code, const std::string &right = std::string())
    : code_(code),
      right_(right) {}
  StatusBuilder(StatusBuilder &&b) noexcept
    : code_(b.code_),
      right_(std::move(b.right_)),
      os_(std::move(b.os_)) {}
  
  StatusBuilder(StatusBuilder &) = delete;

  template <typename T>
  StatusBuilder &&operator<<(const T &value) {
    if (!os_) {
      os_ = std::make_unique<std::ostringstream>();
    }
    (*os_) << value;
    return std::move(*this);
  }

  operator Status() {
    std::string what;
    if (os_) {
      if (!right_.empty()) {
        (*os_) << ": " << right_;
      }
      what = os_->str();
    } else {
      what = right_;
    }

    return Status(code_, what);
  }

 private:
  StatusCode code_;
  std::string right_;
  std::unique_ptr<std::ostringstream> os_;
};

class StatusWrapper;

// store either expected pointer of type T, or a Status which indicates an error
// state.
template<class T>
class expected_ptr {
 public:
  template<class U> friend class expected_ptr;

  expected_ptr(Status &&status) : status_(std::move(status)) {
    ASSERT(!status_.ok());
  }
  expected_ptr(std::unique_ptr<T> &&ptr)
    : ptr_(std::move(ptr)), status_(OkStatus()) {}
  expected_ptr(StatusBuilder &&s) : status_(std::move(s)) {
    ASSERT(!status_.ok());
  }
  expected_ptr(StatusWrapper &&s) : status_(std::move(s)) {
    ASSERT(!status_.ok());
  }

  template<class U>
  expected_ptr(expected_ptr<U> &&s) 
    : status_(std::move(s.status_)),
      ptr_(std::move(s.ptr_)) {}

  expected_ptr &operator=(Status &&status) {
    ASSERT(!status.ok());
    ptr_ = nullptr;
    status_ = std::move(status);
    return *this;
  }
  expected_ptr &operator=(std::unique_ptr<T> &&ptr) {
    ptr_ = std::move(ptr);
    status_ = OkStatus();
    return *this;
  }
  expected_ptr &operator=(expected_ptr &&s) {
    ptr_ = std::move(s.ptr_);
    status_ = std::move(s.status_);
    return *this;
  }

  expected_ptr(expected_ptr &status_or_ptr) = delete;
  expected_ptr &operator=(expected_ptr &s) = delete;

  // methods for Status
  bool ok() const { return status_.ok(); }
  std::string what() const { return status_.what(); }
  StatusCode code() const { return status_.code(); }

  // methods for std::unique_ptr<T>, requires status_.ok()
  T *get() {
    ASSERT(status_.ok());
    return ptr_.get();
  }
  const T *get() const {
    ASSERT(status_.ok());
    return ptr_.get();
  }
  const T &operator*() const & {
    ASSERT(status_.ok());
    return *ptr_;
  }
  T &operator*() & {
    ASSERT(status_.ok());
    return *ptr_;
  }
  const T &&operator*() const && {
    ASSERT(status_.ok());
    return std::move(*ptr_);
  }
  T &&operator*() && {
    ASSERT(status_.ok());
    return std::move(*ptr_);
  }
  const T *operator->() const {
    ASSERT(status_.ok());
    return ptr_.get();
  }
  T* operator->() {
    ASSERT(status_.ok());
    return ptr_.get();
  }

  // convertion
  Status &&status() && {
    return std::move(status_);
  }
  const Status &status() const & {
    return status_;
  }
  std::unique_ptr<T> unique_ptr() && {
    ASSERT(status_.ok());
    return std::move(ptr_);
  }
  std::shared_ptr<T> shared_ptr() && {
    ASSERT(status_.ok());
    return std::move(ptr_);
  }

 private:
  std::unique_ptr<T> ptr_;
  Status status_;
};

class StatusWrapper {
 public:
  StatusWrapper(Status &&status) : status_(std::move(status)) {}
  StatusWrapper(const Status &status) : status_(status.Copy()) {}

  template<class T>
  StatusWrapper(const expected_ptr<T> &status) : status_(StatusCode::kOK) {
    if (!status.ok()) {
      status_ = status.status().Copy();
    }
  }

  StatusWrapper(StatusWrapper &) = delete;
  StatusWrapper operator=(StatusWrapper &) = delete;

  operator Status() && { return std::move(status_); }
  operator bool() { return status_.ok(); }

  template <typename T>
  StatusBuilder operator<<(const T &value) {
    return std::move(StatusBuilder(status_.code(), status_.what()) << value);
  }

 private:
  Status status_;
};

Status OkStatus();
Status OutOfRangeError();
Status AbortedError();
Status AbortedError(const std::string &message);
bool IsOutOfRange(const Status &status);
bool IsOK(const Status &status);

}  // namespace llama

#endif  // LLAMA_RUNTIME_STATUS_H_
