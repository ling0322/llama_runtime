#include "test_helper.h"
#include "path.h"

using namespace llama;


Path _P(const char *pcs) {
  std::string s = pcs;
#ifdef LL_PLATFORM_WINDOWS
  for (char &ch : s) {
    if (ch == '/') ch = '\\';
  }
#endif

  return s;
}

TEST_CASE("get current module path success", "[core][util]") {
  Path current_module_path = Path::CurrentModulePath();
  REQUIRE(!current_module_path.string().empty());
}

TEST_CASE("get current executable path success", "[core][util]") {
  Path current_module_path = Path::CurrentExecutablePath();
  REQUIRE(!current_module_path.string().empty());
}

TEST_CASE("path operations works", "[core][util]") {
  REQUIRE(_P("foo") / _P("bar.txt") == _P("foo/bar.txt"));
  REQUIRE(_P("foo/") / _P("bar.txt") == _P("foo/bar.txt"));
  REQUIRE(_P("foo//") / _P("bar.txt") == _P("foo/bar.txt"));
  REQUIRE(_P("foo") / _P("/bar.txt") == _P("foo/bar.txt"));
  REQUIRE(_P("foo") / _P("//bar.txt") == _P("foo/bar.txt"));
  REQUIRE(_P("foo//") / _P("//bar.txt") == _P("foo/bar.txt"));
  REQUIRE(_P("foo//") / _P("") == _P("foo/"));
  REQUIRE(_P("") / _P("bar.txt") == _P("bar.txt"));
  REQUIRE(_P("") / _P("/bar.txt") == _P("/bar.txt"));

  REQUIRE(_P("foo/bar.txt").basename() == _P("bar.txt"));
  REQUIRE(_P("foo/bar.txt").dirname() == _P("foo"));
  REQUIRE(_P("baz/foo/bar.txt").dirname() == _P("baz/foo"));
  REQUIRE(_P("bar.txt").dirname() == _P(""));
  REQUIRE(_P("foo/").basename() == _P(""));
  REQUIRE(_P("foo//").basename() == _P(""));

  REQUIRE(_P("").basename() == _P(""));
  REQUIRE(_P("").dirname() == _P(""));
}
