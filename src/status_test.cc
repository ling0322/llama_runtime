#include <iostream>
#include <memory>
#include "common.h"
#include "status.h"
#include "test_helper.h"

using namespace llama;

TEST_CASE("status macros works", "[core][common]") {
  Status status = []() -> Status {
    RETURN_IF_ERROR(AbortedError("foo"));
    return OkStatus();
  }();

  REQUIRE(status.code() == StatusCode::kAborted);
  REQUIRE(status.what() == "foo");

  status = []() -> Status {
    RETURN_IF_ERROR(AbortedError("foo")) << "bar";
    return OkStatus();
  }();

  REQUIRE(status.code() == StatusCode::kAborted);
  REQUIRE(status.what() == "bar: foo");
}

TEST_CASE("status move works", "[core][common]") {
  uintptr_t rep = 0;
  Status status = [&rep]() -> Status {
    Status status_internal = AbortedError("foo");
    rep = status_internal.rep();
    RETURN_IF_ERROR(std::move(status_internal));
    return OkStatus();
  }();

  REQUIRE(rep == status.rep());
}

TEST_CASE("status_or works", "[core][common]") {
  uintptr_t rep = 0;
  expected_ptr<int> status_or = [&rep]() -> expected_ptr<int> {
    Status status_internal = AbortedError("foo");
    rep = status_internal.rep();
    RETURN_IF_ERROR(std::move(status_internal));

    return std::make_unique<int>(22);
  }();

  Status status = std::move(status_or).status();
  REQUIRE(rep == status.rep());

  status_or = [&rep]() -> expected_ptr<int> {
    return std::make_unique<int>(22);
  }();
  REQUIRE(status_or.ok());
  REQUIRE(*status_or == 22);
}
