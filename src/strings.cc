#include "strings.h"

#include <stdlib.h>
#include <algorithm>
#include <iomanip>
#include <vector>
#include "common.h"
#include "third_party/utfcpp/utfcpp.h"

namespace llama {
namespace str {

namespace {

template<typename I>
I findFirstNotMatch(I begin, I end, PCStrType chars) {
  I it = begin;
  for (; it < end; ++it) {
    PCStrType pch = chars;
    for (; *pch; ++pch) {
      if (*it == *pch) {
        break;
      }
    }

    if (*pch == '\0') {
      break;
    }
  }

  return it;
}

}  // namespace



#if !defined(WCHAR_MAX)
#error WCHAR_MAX not defined!
#endif
#if defined(_MSC_VER) && _MSC_VER <= 1310
#define AL_INTERNAL_WCHAR_IS_UTF16
#elif WCHAR_MAX > 0x10000
#define AL_INTERNAL_WCHAR_IS_UTF32
#else
#define AL_INTERNAL_WCHAR_IS_UTF16
#endif

std::string toUtf8(const std::u16string &u16s) {
  std::string s;
  utf8::utf16to8(u16s.begin(), u16s.end(), std::back_inserter(s));
  return s;
}

std::u16string toUtf16(const std::string &s) {
  std::u16string u16s;
  utf8::utf8to16(s.begin(), s.end(), std::back_inserter(u16s));
  return u16s;
}

std::string toUtf8(const std::u32string &u32s) {
  std::string s;
  utf8::utf32to8(u32s.begin(), u32s.end(), std::back_inserter(s));
  return s;
}

std::u32string toUtf32(const std::string &s) {
  std::u32string u32s;
  utf8::utf8to32(s.begin(), s.end(), std::back_inserter(u32s));
  return u32s;
}

std::wstring toWide(const std::string &s) {
  std::wstring ws;

#if defined(AL_INTERNAL_WCHAR_IS_UTF32)
  utf8::utf8to32(s.begin(), s.end(), std::back_inserter(ws));
#elif defined(AL_INTERNAL_WCHAR_IS_UTF16)
  utf8::utf8to16(s.begin(), s.end(), std::back_inserter(ws));
#else
#error macro AL_INTERNAL_WCHAR_IS_ not defined
#endif

  return ws;
}

std::string ToUtf8(const std::wstring &ws) {
  std::string s;

#if defined(AL_INTERNAL_WCHAR_IS_UTF32)
  utf8::utf32to8(ws.begin(), ws.end(), std::back_inserter(s));
#elif defined(AL_INTERNAL_WCHAR_IS_UTF16)
  utf8::utf16to8(ws.begin(), ws.end(), std::back_inserter(s));
#else
#error macro AL_INTERNAL_WCHAR_IS_ not defined
#endif

  return s;
}

std::string trimLeft(const std::string &s, PCStrType chars) {
  auto it = findFirstNotMatch(s.begin(), s.end(), chars);
  return std::string(it, s.end());
}

std::string trimRight(const std::string &s, PCStrType chars) {
  auto it = findFirstNotMatch(s.rbegin(), s.rend(), chars);
  auto n_deleted = it - s.rbegin();
  return std::string(s.begin(), s.end() - n_deleted);
}

std::string trim(const std::string &s, PCStrType chars) {
  auto it_begin = findFirstNotMatch(s.begin(), s.end(), chars);
  if (it_begin == s.end()) {
    return "";
  }

  auto it_r = findFirstNotMatch(s.rbegin(), s.rend(), chars);
  auto n_deleted = it_r - s.rbegin();
  auto it_end = s.end() - n_deleted;

  ASSERT(it_end > it_begin);
  return std::string(it_begin, it_end);
}

std::vector<std::string> split(const std::string &str, const std::string &delim) {
  std::vector<std::string> fields;
  int start = 0;
  int pos = 0;
  while ((pos = str.find(delim, start)) != std::string::npos) {
    fields.emplace_back(str.cbegin() + start, str.cbegin() + pos);
    start = pos + delim.size();
  }
  
  fields.emplace_back(str.cbegin() + start, str.cend());
  return fields;
}

std::string toLower(const std::string &s) {
  std::string lower(s.begin(), s.end());
  std::transform(lower.begin(), lower.end(), lower.begin(), tolower);
  return lower;
}

int atoi(const std::string &s) {
  char *p = nullptr;
  long v = strtol(s.c_str(), &p, 0);
  if (*p == '\0') {
    return static_cast<int>(v);
  } else {
    throw AbortedException(str::sprintf("invalid integer string: %s", s));
  }
}

int _Sprintf0_ReadDigit(const char **ppch, char *buf, int buf_size) {
  const char *pch = *ppch;
  char *pbuf = buf;

  ASSERT(isdigit(*pch));
  *pbuf = *pch;
  ++pbuf;
  ++pch;

  while (isdigit(*pch)) {
    int digit_len = static_cast<int>(pbuf - buf);
    if (digit_len >= buf_size - 1) {
      return kSprintfMaxWeight;
    }

    *pbuf = *pch;
    ++pbuf;
    ++pch;
  }

  *pbuf = '\0';
  *ppch = pch;

  int n = atoi(buf);
  return n < kSprintfMaxWeight ? n : kSprintfMaxWeight;
}

// parse format string and apply to ss
char sprintf0_parseFormat(const char **pp_string, std::stringstream &ss) {
  char digit_buffer[32];
  std::string format_string;
  const char *pch = *pp_string;

  ASSERT(*pch == '%');
  format_string.push_back(*pch);
  ++pch;

  // %
  if (*pch == '%') {
    *pp_string = ++pch;
    return '%';
  }

  // flags
  switch (*pch) {
    case '-':
      ss << std::left;
      ++pch;
      break;
    case '+':
      ss << std::showpos;
      ++pch;
      break;
    case ' ':
      ++pch;
      break;
    case '0':
      ss << std::setfill('0');
      ++pch;
      break;
  }

  // width
  if (isdigit(*pch)) {
    int n = _Sprintf0_ReadDigit(&pch, digit_buffer, sizeof(digit_buffer));
    ss << std::setw(n);
  }

  // precision
  if (*pch == '.') {
    ++pch;

    if (isdigit(*pch)) {
      int n = _Sprintf0_ReadDigit(&pch, digit_buffer, sizeof(digit_buffer));
      ss << std::setprecision(n);
    } else {
      *pp_string = ++pch;
      return '#';
    }
  }

  // specifier
  char type_specifier = *pch;
  switch (*pch) {
    case 'd':
    case 'i':
    case 'u':
      ss << std::dec;
      break;
    case 'x':
    case 'p':
      ss << std::hex;
      break;
    case 'X':
      ss << std::hex << std::uppercase;
      break;
    case 'e':
      ss << std::scientific;
      break;
    case 'E':
      ss << std::scientific << std::uppercase;
      break;
    case 'g':
      ss << std::defaultfloat;
      break;
    case 'G':
      ss << std::defaultfloat << std::uppercase;
      break;
    case 'a':
      ss << std::hexfloat;
      break;
    case 'A':
      ss << std::hexfloat << std::uppercase;
      break;
    case 'f':
      ss << std::fixed;
      break;
    case 's':
    case 'c':
      break;
    default:
      *pp_string = *pch ? pch + 1 : pch;
      return '#';
  }
  ++pch;

  *pp_string = pch;
  return type_specifier;
}

std::string replace(const std::string &from, const std::string &old, const std::string &repl) {
  int pos = 0;
  std::string s = from;
  while((pos = s.find(old, pos)) != std::string::npos) {
    s.replace(pos, old.length(), repl);
    pos += repl.length();
  }

  return s;
}

std::vector<std::string> splitToUtf8Chars(const std::string &s) {
  std::vector<std::string> utf8Chars;

  std::string::const_iterator begin = s.begin();
  uint32_t cp = 0;
  char singleChar[] = " ";
  while (begin < s.end()) {
    std::string::const_iterator next = begin;
    try {
      uint32_t cp = utf8::next(next, s.end());
      utf8Chars.emplace_back(begin, next);
      begin = next;
    } catch (const AbortedException &) {
      singleChar[0] = *begin;
      utf8Chars.emplace_back(singleChar);
      ++begin;
    }
  }

  return utf8Chars;
}

}  // namespace strings
}  // namespace llama
