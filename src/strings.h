#ifndef LLAMA_CC_STRINGS_H_
#define LLAMA_CC_STRINGS_H_

#include <string>
#include <type_traits>
#include <vector>
#include "common.h"
#include "status.h"

namespace llama {
namespace strings {

Status ToUtf8(const std::u16string &u16s, std::string *s);
Status ToUtf8(const std::wstring &ws, std::string *s);
Status ToUtf8(const std::u32string &u32s, std::string *s);
Status ToUtf16(const std::string &s, std::u16string *u16s);
Status ToUtf32(const std::string &s, std::u32string *u32s);
Status ToWide(const std::string &s, std::wstring *ws);

std::string TrimLeft(const std::string &s, PCStrType chars = " \t\r\n");
std::string TrimRight(const std::string &s, PCStrType chars = " \t\r\n");
std::string Trim(const std::string &s, PCStrType chars = " \t\r\n");
std::vector<std::string> Split(const std::string &str,
                         const std::string &delim);

std::string Replace(const std::string &s,
                    const std::string &old,
                    const std::string &repl);

// split a utf8 string into a list of strings. Each string in this list only
// contains one character in utf-8 encoding. For invalid byte, it will keep
// it as a standalone char.
std::vector<std::string> SplitToUtf8Chars(const std::string &s);

// internal functions for Sprintf()
constexpr int kSprintfMaxWeight = 200;
char _Sprintf0_ParseFormat(const char **pp_string, std::stringstream &ss);
template<typename T>
bool _Sprintf0_CheckType(char type_specifier) {
  switch (type_specifier) {
    case 'd':
    case 'i':
    case 'u':
    case 'x':
    case 'X':
      return std::is_integral<std::decay<T>::type>::value &&
             !std::is_same<std::decay<T>::type, char>::value;
    case 'p':
      return std::is_pointer<std::decay<T>::type>::value;
    case 'e':
    case 'E':
    case 'g':
    case 'G':
    case 'a':
    case 'A':
    case 'f':
      return std::is_floating_point<std::decay<T>::type>::value;
    case 's':
      return std::is_same<std::decay<T>::type, std::string>::value ||
             std::is_same<std::decay<T>::type, char *>::value ||
             std::is_same<std::decay<T>::type, const char *>::value;
    case 'c':
      return std::is_same<std::decay<T>::type, char>::value;
    case '#':
      return false;
  }

  return true;
}
inline std::string _Sprintf0(std::stringstream &ss, const char *pch) {
  while (*pch) {
    if (*pch == '%') {
      char type_specifier = _Sprintf0_ParseFormat(&pch, ss);
      if (type_specifier != '%') {
        ss << "%!" << type_specifier << "(<null>)";
      } else {
        ss << '%';
      }
    } else {
      ss << *pch++;
    }
  }
  return ss.str();
}
template<typename T, typename... Args>
inline std::string _Sprintf0(std::stringstream &ss, const char *pch,
                             T &&value, Args &&...args) {
  const auto default_precision = ss.precision();
  const auto default_width = ss.width();
  const auto default_flags = ss.flags();
  const auto default_fill = ss.fill();

  while (*pch != '%' && *pch != '\0') {
    ss << *pch++;
  }

  bool type_correct;
  char type_specifier;
  const char *pch_fmtb = pch;
  if (*pch) {
    type_specifier = _Sprintf0_ParseFormat(&pch, ss);
    if (type_specifier == '%') {
      ss << '%';
      return _Sprintf0(ss, pch, std::forward<T &&>(value),
                      std::forward<Args>(args)...);
    }
    type_correct = _Sprintf0_CheckType<T>(type_specifier);
    if (type_correct) {
      ss << std::move(value);
    }
  } else {
    type_specifier = '_';
    type_correct = false;
  }

  ss.setf(default_flags);
  ss.precision(default_precision);
  ss.width(default_width);
  ss.fill(default_fill);

  if (type_specifier == '#') {
    pch_fmtb++;
    ss << "%!" << std::string(pch_fmtb, pch) <<  "(" << std::move(value) << ")";
  } else if (!type_correct) {
    ss << "%!" << type_specifier << "(" << std::move(value) << ")";
  }

  return _Sprintf0(ss, pch, std::forward<Args>(args)...);
}

// String formatting, for example:
//   util::Sprintf("%s %d", "foo", 233);
template<typename... Args>
inline std::string Sprintf(const std::string &fmt, Args &&...args) {
  return _Sprintf0(std::stringstream(), fmt.c_str(), std::forward<Args>(args)...);
}

}  // namespace strings
}  // namespace llama

#endif  // LLAMA_CC_STRINGS_H_
