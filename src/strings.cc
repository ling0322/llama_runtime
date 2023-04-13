#include "strings.h"

#include <stdlib.h>
#include <iomanip>
#include <vector>
#include "common.h"
#include "status.h"
#include "third_party/utfcpp/utfcpp.h"

namespace llama {
namespace strings {

namespace {

template<typename I>
I FindFirstNotMatch(I begin, I end, PCStrType chars) {
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
#define BR_INTERNAL_WCHAR_IS_UTF16
#elif WCHAR_MAX > 0x10000
#define BR_INTERNAL_WCHAR_IS_UTF32
#else
#define BR_INTERNAL_WCHAR_IS_UTF16
#endif

Status ToUtf8(const std::u16string &u16s, std::string *s) {
  s->clear();
  RETURN_IF_ERROR(utf8::utf16to8(
      u16s.begin(),
      u16s.end(),
      std::back_inserter(*s)));
  return OkStatus();
}

Status ToUtf16(const std::string &s, std::u16string *u16s) {
    u16s->clear();
    RETURN_IF_ERROR(utf8::utf8to16(
        s.begin(),
        s.end(),
        std::back_inserter(*u16s)));
    return OkStatus();
}

Status ToUtf8(const std::u32string &u32s, std::string *s) {
  s->clear();
  RETURN_IF_ERROR(utf8::utf32to8(
      u32s.begin(),
      u32s.end(),
      std::back_inserter(*s)));
  return OkStatus();
}

Status ToUtf32(const std::string &s, std::u32string *u32s) {
  u32s->clear();
  RETURN_IF_ERROR(utf8::utf8to32(
      s.begin(),
      s.end(),
      std::back_inserter(*u32s)));
  return OkStatus();
}

Status ToWide(const std::string &s, std::wstring *ws) {
  ws->clear();

#if defined(BR_INTERNAL_WCHAR_IS_UTF32)
  RETURN_IF_ERROR(utf8::utf8to32(
      s.begin(),
      s.end(),
      std::back_inserter(*ws)));
#elif defined(BR_INTERNAL_WCHAR_IS_UTF16)
  RETURN_IF_ERROR(utf8::utf8to16(
      s.begin(),
      s.end(),
      std::back_inserter(*ws)));
#else
#error macro BR_INTERNAL_WCHAR_IS_ not defined
#endif

  return OkStatus();
}

Status ToUtf8(const std::wstring &ws, std::string *s) {
  s->clear();

#if defined(BR_INTERNAL_WCHAR_IS_UTF32)
  RETURN_IF_ERROR(utf8::utf32to8(
      ws.begin(),
      ws.end(),
      std::back_inserter(*s)));
#elif defined(BR_INTERNAL_WCHAR_IS_UTF16)
  RETURN_IF_ERROR(utf8::utf16to8(
      ws.begin(),
      ws.end(),
      std::back_inserter(*s)));
#else
#error macro BR_INTERNAL_WCHAR_IS_ not defined
#endif

  return OkStatus();
}

std::string TrimLeft(const std::string &s, PCStrType chars) {
  auto it = FindFirstNotMatch(s.begin(), s.end(), chars);
  return std::string(it, s.end());
}

std::string TrimRight(const std::string &s, PCStrType chars) {
  auto it = FindFirstNotMatch(s.rbegin(), s.rend(), chars);
  auto n_deleted = it - s.rbegin();
  return std::string(s.begin(), s.end() - n_deleted);
}

std::string Trim(const std::string &s, PCStrType chars) {
  auto it_begin = FindFirstNotMatch(s.begin(), s.end(), chars);
  if (it_begin == s.end()) {
    return "";
  }

  auto it_r = FindFirstNotMatch(s.rbegin(), s.rend(), chars);
  auto n_deleted = it_r - s.rbegin();
  auto it_end = s.end() - n_deleted;

  ASSERT(it_end > it_begin);
  return std::string(it_begin, it_end);
}

std::vector<std::string> Split(
    const std::string &str,
    const std::string &delim) {
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
char _Sprintf0_ParseFormat(const char **pp_string, std::stringstream &ss) {
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
      ss << std::hex << std::showbase;
      break;
    case 'X':
      ss << std::hex << std::showbase << std::uppercase;
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

}  // namespace strings
}  // namespace llama
