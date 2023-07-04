#include "test_helper.h"
#include "strings.h"

using namespace llama;
using namespace strings;

std::vector<std::string> V(std::initializer_list<std::string> il) {
  return il;
}

TEST_CASE("string functions works", "[core][util]") {
  REQUIRE(Trim("  ") == "");
  REQUIRE(Trim(" \ta ") == "a");
  REQUIRE(Trim("a ") == "a");
  REQUIRE(Trim("a\t") == "a");
  REQUIRE(Trim("a") == "a");

  REQUIRE(TrimLeft(" \t") == "");
  REQUIRE(TrimLeft(" \ta") == "a");
  REQUIRE(TrimLeft(" \ta ") == "a ");
  REQUIRE(TrimLeft("a ") == "a ");

  REQUIRE(TrimRight(" \t") == "");
  REQUIRE(TrimRight("a\t ") == "a");
  REQUIRE(TrimRight(" \ta\t ") == " \ta");
  REQUIRE(TrimRight(" a") == " a");
  REQUIRE(TrimRight("a") == "a");

  REQUIRE(Split("A\tB\tC", "\t") == V({"A", "B", "C"}));
  REQUIRE(Split("A.-B.-C", ".-") == V({"A", "B", "C"}));
  REQUIRE(Split("A.B.C.", ".") == V({"A", "B", "C", ""}));
  REQUIRE(Split("..A.B", ".") == V({"", "", "A", "B"}));

  std::string s, s_ref = "vanilla\xe5\x87\xaa\xc3\xa2\xf0\x9f\x8d\xad";
  std::wstring ws, ws_ref = L"vanilla\u51ea\u00e2\U0001f36d";
  REQUIRE(ToWide(s_ref, &ws).ok());
  REQUIRE(ws == ws_ref);

  REQUIRE(ToUtf8(ws_ref, &s).ok());
  REQUIRE(s == s_ref);
}

TEST_CASE("sprintf works", "[core][util]") {
  // BVT
  REQUIRE(Sprintf("%d", 22) == "22");
  REQUIRE(Sprintf("foo_%d", 22) == "foo_22");
  REQUIRE(Sprintf("foo%d %s", 22, "33") == "foo22 33");

  // integer
  int i = 1234567;
  REQUIRE(Sprintf("%010d", i) == "0001234567");
  REQUIRE(Sprintf("%10d", i) == "   1234567");
  REQUIRE(Sprintf("%x", i) == "0x12d687");
  REQUIRE(Sprintf("%10x", i) == "  0x12d687");
  REQUIRE(Sprintf("%X", i) == "0X12D687");

  // float
  double f = 123.4567;
  double g = 1.234567e8;
  REQUIRE(Sprintf("%.6f", f) == "123.456700");
  REQUIRE(Sprintf("%.3f", f) == "123.457");
  REQUIRE(Sprintf("%9.2f", f) == "   123.46");
  REQUIRE(Sprintf("%09.2f", f) == "000123.46");
  REQUIRE(Sprintf("%.3e", f) == "1.235e+02");
  REQUIRE(Sprintf("%.3E", f) == "1.235E+02");
  REQUIRE(Sprintf("%.5g", f) == "123.46");
  REQUIRE(Sprintf("%.5g", g) == "1.2346e+08");
  REQUIRE(Sprintf("%.5G", g) == "1.2346E+08");

  // string
  std::string foo = "foo";
  const char* bar = "bar";
  REQUIRE(Sprintf("%s", foo) == "foo");
  REQUIRE(Sprintf("%s", bar) == "bar");
  REQUIRE(Sprintf("%s %s", foo, bar) == "foo bar");
  REQUIRE(Sprintf("%10s", foo) == "       foo");
  REQUIRE(Sprintf("%-10s", foo) == "foo       ");

  // char
  REQUIRE(Sprintf("%c", 'c') == "c");

  // edge cases
  REQUIRE(Sprintf("%10.2f %.3e", f, f) == "    123.46 1.235e+02");
  REQUIRE(Sprintf("%%%.5e", f) == "%1.23457e+02");
  REQUIRE(Sprintf("%%%d%d%d%%", 1, 2, 3) == "%123%");
  REQUIRE(Sprintf("%10000000d", 22) == Sprintf("%200d", 22));
  REQUIRE(Sprintf("%1000000000000d", 22) == Sprintf("%200d", 22));
  REQUIRE(Sprintf("foo") == "foo");
  REQUIRE(Sprintf("%%") == "%");
  REQUIRE(Sprintf("") == "");

  // invalid format string
  REQUIRE(Sprintf("%s_%d", "foo") == "foo_%!d(<null>)");
  REQUIRE(Sprintf("%s", "foo", 22) == "foo%!_(22)");
  REQUIRE(Sprintf("%d", "foo") == "%!d(foo)");
  REQUIRE(Sprintf("%d_foo_%d_0", 22) == "22_foo_%!d(<null>)_0");
  REQUIRE(Sprintf("%o", 22) == "%!o(22)");
  REQUIRE(Sprintf("%8.3o", 22) == "%!8.3o(22)");
  REQUIRE(Sprintf("%8", 22) == "%!8(22)");
  REQUIRE(Sprintf("%8%", 22) == "%!8%(22)");
  REQUIRE(Sprintf("%") == "%!#(<null>)");
  REQUIRE(Sprintf("%", 22) == "%!(22)");
  REQUIRE(Sprintf("%", 22, "foo") == "%!(22)%!_(foo)");
  REQUIRE(Sprintf("%8.ad", 22) == "%!8.a(22)d");
}
