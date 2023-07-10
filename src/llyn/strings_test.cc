#include "catch2/catch_amalgamated.hpp"
#include "llyn/strings.h"

using namespace ly;

std::vector<std::string> V(std::initializer_list<std::string> il) {
  return il;
}

TEST_CASE("string functions works", "[core][util]") {
  REQUIRE(trim("  ") == "");
  REQUIRE(trim(" \ta ") == "a");
  REQUIRE(trim("a ") == "a");
  REQUIRE(trim("a\t") == "a");
  REQUIRE(trim("a") == "a");

  REQUIRE(trimLeft(" \t") == "");
  REQUIRE(trimLeft(" \ta") == "a");
  REQUIRE(trimLeft(" \ta ") == "a ");
  REQUIRE(trimLeft("a ") == "a ");

  REQUIRE(trimRight(" \t") == "");
  REQUIRE(trimRight("a\t ") == "a");
  REQUIRE(trimRight(" \ta\t ") == " \ta");
  REQUIRE(trimRight(" a") == " a");
  REQUIRE(trimRight("a") == "a");

  REQUIRE(split("A\tB\tC", "\t") == V({"A", "B", "C"}));
  REQUIRE(split("A.-B.-C", ".-") == V({"A", "B", "C"}));
  REQUIRE(split("A.B.C.", ".") == V({"A", "B", "C", ""}));
  REQUIRE(split("..A.B", ".") == V({"", "", "A", "B"}));

  std::string s, s_ref = "vanilla\xe5\x87\xaa\xc3\xa2\xf0\x9f\x8d\xad";
  std::wstring ws, ws_ref = L"vanilla\u51ea\u00e2\U0001f36d";
  REQUIRE(toWide(s_ref) == ws_ref);
  REQUIRE(toUtf8(ws_ref) == s_ref);
}

TEST_CASE("sprintf works", "[core][util]") {
  // BVT
  REQUIRE(ly::sprintf("%d", 22) == "22");
  REQUIRE(ly::sprintf("foo_%d", 22) == "foo_22");
  REQUIRE(ly::sprintf("foo%d %s", 22, "33") == "foo22 33");

  // integer
  int i = 1234567;
  REQUIRE(ly::sprintf("%010d", i) == "0001234567");
  REQUIRE(ly::sprintf("%10d", i) == "   1234567");
  REQUIRE(ly::sprintf("%x", i) == "12d687");
  REQUIRE(ly::sprintf("%10x", i) == "    12d687");
  REQUIRE(ly::sprintf("%X", i) == "12D687");

  // float
  double f = 123.4567;
  double g = 1.234567e8;
  REQUIRE(ly::sprintf("%.6f", f) == "123.456700");
  REQUIRE(ly::sprintf("%.3f", f) == "123.457");
  REQUIRE(ly::sprintf("%9.2f", f) == "   123.46");
  REQUIRE(ly::sprintf("%09.2f", f) == "000123.46");
  REQUIRE(ly::sprintf("%.3e", f) == "1.235e+02");
  REQUIRE(ly::sprintf("%.3E", f) == "1.235E+02");
  REQUIRE(ly::sprintf("%.5g", f) == "123.46");
  REQUIRE(ly::sprintf("%.5g", g) == "1.2346e+08");
  REQUIRE(ly::sprintf("%.5G", g) == "1.2346E+08");

  // string
  std::string foo = "foo";
  const char* bar = "bar";
  REQUIRE(ly::sprintf("%s", foo) == "foo");
  REQUIRE(ly::sprintf("%s", bar) == "bar");
  REQUIRE(ly::sprintf("%s %s", foo, bar) == "foo bar");
  REQUIRE(ly::sprintf("%10s", foo) == "       foo");
  REQUIRE(ly::sprintf("%-10s", foo) == "foo       ");

  // char
  REQUIRE(ly::sprintf("%c", 'c') == "c");

  // edge cases
  REQUIRE(ly::sprintf("%10.2f %.3e", f, f) == "    123.46 1.235e+02");
  REQUIRE(ly::sprintf("%%%.5e", f) == "%1.23457e+02");
  REQUIRE(ly::sprintf("%%%d%d%d%%", 1, 2, 3) == "%123%");
  REQUIRE(ly::sprintf("%10000000d", 22) == ly::sprintf("%200d", 22));
  REQUIRE(ly::sprintf("%1000000000000d", 22) == ly::sprintf("%200d", 22));
  REQUIRE(ly::sprintf("foo") == "foo");
  REQUIRE(ly::sprintf("%%") == "%");
  REQUIRE(ly::sprintf("") == "");

  // invalid format string
  REQUIRE(ly::sprintf("%s_%d", "foo") == "foo_%!d(<null>)");
  REQUIRE(ly::sprintf("%s", "foo", 22) == "foo%!_(22)");
  REQUIRE(ly::sprintf("%d", "foo") == "%!d(foo)");
  REQUIRE(ly::sprintf("%d_foo_%d_0", 22) == "22_foo_%!d(<null>)_0");
  REQUIRE(ly::sprintf("%o", 22) == "%!o(22)");
  REQUIRE(ly::sprintf("%8.3o", 22) == "%!8.3o(22)");
  REQUIRE(ly::sprintf("%8", 22) == "%!8(22)");
  REQUIRE(ly::sprintf("%8%", 22) == "%!8%(22)");
  REQUIRE(ly::sprintf("%") == "%!#(<null>)");
  REQUIRE(ly::sprintf("%", 22) == "%!(22)");
  REQUIRE(ly::sprintf("%", 22, "foo") == "%!(22)%!_(foo)");
  REQUIRE(ly::sprintf("%8.ad", 22) == "%!8.a(22)d");
}
