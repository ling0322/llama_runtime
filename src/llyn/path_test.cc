#include "catch2/catch_amalgamated.hpp"
#include "llyn/path.h"
#include "llyn/platform.h"

using namespace ly;

Path toPath(const char *pcs) {
  std::string s = pcs;
#ifdef LY_PLATFORM_WINDOWS
  for (char &ch : s) {
    if (ch == '/') ch = '\\';
  }
#endif

  return s;
}

TEST_CASE("get current module path success", "[core][util]") {
  Path current_module_path = Path::currentModulePath();
  REQUIRE(!current_module_path.string().empty());
}

TEST_CASE("get current executable path success", "[core][util]") {
  Path current_module_path = Path::currentExecutablePath();
  REQUIRE(!current_module_path.string().empty());
}

TEST_CASE("path operations works", "[core][util]") {
  REQUIRE(toPath("foo") / toPath("bar.txt") == toPath("foo/bar.txt"));
  REQUIRE(toPath("foo/") / toPath("bar.txt") == toPath("foo/bar.txt"));
  REQUIRE(toPath("foo//") / toPath("bar.txt") == toPath("foo/bar.txt"));
  REQUIRE(toPath("foo") / toPath("/bar.txt") == toPath("foo/bar.txt"));
  REQUIRE(toPath("foo") / toPath("//bar.txt") == toPath("foo/bar.txt"));
  REQUIRE(toPath("foo//") / toPath("//bar.txt") == toPath("foo/bar.txt"));
  REQUIRE(toPath("foo//") / toPath("") == toPath("foo/"));
  REQUIRE(toPath("") / toPath("bar.txt") == toPath("bar.txt"));
  REQUIRE(toPath("") / toPath("/bar.txt") == toPath("/bar.txt"));

  REQUIRE(toPath("foo/bar.txt").basename() == toPath("bar.txt"));
  REQUIRE(toPath("foo/bar.txt").dirname() == toPath("foo"));
  REQUIRE(toPath("baz/foo/bar.txt").dirname() == toPath("baz/foo"));
  REQUIRE(toPath("bar.txt").dirname() == toPath(""));
  REQUIRE(toPath("foo/").basename() == toPath(""));
  REQUIRE(toPath("foo//").basename() == toPath(""));

  REQUIRE(toPath("").basename() == toPath(""));
  REQUIRE(toPath("").dirname() == toPath(""));
}
