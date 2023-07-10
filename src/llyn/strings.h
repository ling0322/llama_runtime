#pragma once

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "llyn/internal/sprintf.h"

namespace ly {

std::string toUtf8(const std::u16string &u16s);
std::string toUtf8(const std::wstring &ws);
std::string toUtf8(const std::u32string &u32s);
std::u16string toUtf16(const std::string &s);
std::u32string toUtf32(const std::string &s);
std::wstring toWide(const std::string &s);

std::string trimLeft(const std::string &s, const char *chars = " \t\r\n");
std::string trimRight(const std::string &s, const char *chars = " \t\r\n");
std::string trim(const std::string &s, const char *chars = " \t\r\n");
std::vector<std::string> split(const std::string &str, const std::string &delim);

std::string replace(const std::string &s, const std::string &old, const std::string &repl);
std::string toLower(const std::string &s);

// string to int. throw AbortedException if parsing failed.
int atoi(const std::string &s);

// split s string into utf-8 characters (string),
std::vector<std::string> splitUtf8(const std::string &s);

// String formatting, for example:
//   ly::sprintf("%s %d", "foo", 233);
template<typename... Args>
inline std::string sprintf(const std::string &fmt, Args &&...args) {
  return internal::sprintf0(std::stringstream(), fmt.c_str(), std::forward<Args>(args)...);
}

}  // namespace ly
