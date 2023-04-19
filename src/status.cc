#include "status.h"

#include "common.h"
#include <string>

namespace llama {

struct Status::Rep {
  std::string what;
  StatusCode code;
};

Status::Status(StatusCode code) : rep_(static_cast<int>(code) << 1) {}
Status::Status(StatusCode code, const std::string &what) {
  Rep *rep = new Rep();
  rep->code = code;
  rep->what = what;

  rep_ = reinterpret_cast<uintptr_t>(rep) + 1;
}

Status::Status(Status &&status) : rep_(status.rep_) {
  status.rep_ = 0;
}

Status Status::Copy() const {
  Status status(StatusCode::kOK);
  if (IsHeapAllocated()) {
    Rep *from_rep = GetRep();
    Rep *rep = new Rep(*from_rep);

    status.SetRep(rep);
  } else {
    status.rep_ = rep_;
  }

  return status;
}

Status::~Status() {
  if (IsHeapAllocated()) {
    Rep *rep = GetRep();
    delete rep;
  }
  
  rep_ = 0;
}

Status &Status::operator=(Status &&status) {
  rep_ = status.rep_;
  status.rep_ = 0;

  return *this;
}

std::string Status::what() const {
  if (IsHeapAllocated()) {
    return GetRep()->what;
  } else {
    return DefaultMessage(GetCode());
  }
}

StatusCode Status::code() const {
  if (IsHeapAllocated()) {
    return GetRep()->code;
  } else {
    return GetCode();
  }
}

Status::Rep *Status::GetRep() const {
  ASSERT(IsHeapAllocated());
  return reinterpret_cast<Rep *>(rep_ - 1);
}

StatusCode Status::GetCode() const {
  ASSERT(!IsHeapAllocated());
  return static_cast<StatusCode>(rep_ >> 1);
}

void Status::SetRep(Rep *rep) {
  rep_ = reinterpret_cast<uintptr_t>(rep) + 1;
}

std::string Status::DefaultMessage(StatusCode code) const {
  switch (code) {
    case StatusCode::kOK:
      return "OK";
    case StatusCode::kAborted:
      return "aborted";
    case StatusCode::kOutOfRange:
      return "out of range";
    default:
      return "unknown";
  }
}

Status OkStatus() {
  return Status(StatusCode::kOK);
}
Status OutOfRangeError() {
  return Status(StatusCode::kOutOfRange);
}
Status AbortedError() {
  return Status(StatusCode::kAborted);
}
Status AbortedError(const std::string &message) {
  return Status(StatusCode::kAborted, message);
}
bool IsOutOfRange(const Status &status) {
  return status.code() == StatusCode::kOutOfRange;
}
bool IsOK(const Status &status) {
  return status.ok();
}

}  // namespace llama
