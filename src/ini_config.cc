#include "ini_config.h"

#include <tuple>
#include "reader.h"
#include "strings.h"
#include "util.h"

namespace llama {

// -- class IniConfig ----------------------------------------------------------

IniConfig::IniConfig() {}

expected_ptr<IniConfig> IniConfig::Read(const std::string &filename) {
  std::unique_ptr<IniConfig> config{new IniConfig()};

  auto fp = ReadableFile::Open(filename);
  RETURN_IF_ERROR(fp);

  Scanner scanner{fp.get()};

  enum State {
    kBegin,
    kSelfLoop,
  };

  util::Path ini_dir = util::Path(filename).dirname();
  IniSection section{ini_dir};
  State state = kBegin;
  std::string name, key, value;
  while (scanner.Scan()) {
    std::string line = strings::Trim(scanner.text());
    if (state == kBegin) {
      if (IsEmptyLine(line)) {
        // self-loop
      } else if (IsHeader(line)) {
        RETURN_IF_ERROR(ParseHeader(line, &name));
        section.name_ = name;
        state = kSelfLoop;
      } else {
        RETURN_ABORTED() << "invalid line: " << line;
      }
    } else if (state == kSelfLoop) {
      if (IsEmptyLine(line)) {
        // self-loop
      } else if (IsHeader(line)) {
        config->table_.emplace(section.name(), std::move(section));
        section = IniSection{ini_dir};
        RETURN_IF_ERROR(ParseHeader(line, &name));
        section.name_ = name;
      } else {
        RETURN_IF_ERROR(ParseKeyValue(line, &key, &value));
        section.kv_table_[key] = value;
      }
    } else {
      NOT_IMPL();
    }
  }
  
  RETURN_IF_ERROR(scanner.status());
  if (state == kBegin) {
    RETURN_ABORTED() << "ini file is empty.";
  }

  config->table_.emplace(section.name(), std::move(section));

  return config;
}

bool IniConfig::has_section(const std::string &section) const {
  auto it = table_.find(section);
  return it != table_.end();
}

Status IniConfig::EnsureSection(const std::string &name) const {
  auto it = table_.find(name);
  if (it == table_.end()) {
    RETURN_OUT_OF_RANGE() << "section not found: " << name;
  }

  return OkStatus();
}

const IniSection &IniConfig::section(const std::string &name) const {
  auto it = table_.find(name);
  CHECK(it != table_.end()) << "section not found: " << name;

  return it->second;
}

bool IniConfig::IsEmptyLine(const std::string &s) {
  if (s.empty() || s.front() == ';') {
    return true;
  } else {
    return false;
  }
}

bool IniConfig::IsHeader(const std::string &s) {
  if (s.front() == '[' && s.back() == ']') {
    return true;
  } else {
    return false;
  }
}

Status IniConfig::ParseHeader(const std::string &s, std::string *name) {
  if (!IsHeader(s)) {
    RETURN_ABORTED() << "invalid line: "<< s;
  }
  
  *name = s.substr(1, s.size() - 2);
  *name = strings::Trim(*name);
  if (name->empty()) {
    RETURN_ABORTED() << "invalid ini section: "<< s;
  }

  return OkStatus();
}

Status IniConfig::ParseKeyValue(
    const std::string &s,
    std::string *key,
    std::string *value) {
  auto row = strings::Split(s, "=");
  if (row.size() != 2) {
    RETURN_ABORTED() << "invalid line: " << s;
  }
  *key = strings::ToLower(strings::Trim(row[0]));
  *value = strings::Trim(row[1]);
  if (key->empty() || value->empty()) {
    RETURN_ABORTED() << "invalid line: " << s;
  }

  return OkStatus();
}


// -- class IniSection ---------------------------------------------------------

IniSection::IniSection(const util::Path &ini_dir)
    : ini_dir_(ini_dir) {}

template<>
Status IniSection::Get(const std::string &key, std::string *val) const {
  auto it = kv_table_.find(key);
  if (it == kv_table_.end()) {
    RETURN_ABORTED() << "key not found (ini_session=" << name_ << "): " << key;
  }

  *val = it->second;
  return OkStatus();
}

template<>
Status IniSection::Get(const std::string &key, int *val) const {
  std::string s;
  RETURN_IF_ERROR(Get(key, &s));

  int v;
  RETURN_IF_ERROR(strings::Atoi(s, &v));
  *val = v;

  return OkStatus();
}

template<>
Status IniSection::Get(const std::string &key, bool *val) const {
  std::string s;
  RETURN_IF_ERROR(Get(key, &s));

  s = strings::ToLower(s);
  if (s == "true" || s == "1") {
    *val = true;
  } else if (s == "false" || s == "0") {
    *val = false;
  } else {
    RETURN_ABORTED() << "invalid bool value: " << s;
  }

  return OkStatus();
}

template<>
Status IniSection::Get(const std::string &key, util::Path *val) const {
  std::string s;
  RETURN_IF_ERROR(Get(key, &s));

  util::Path path(s);
  if (!path.isabs()) {
    path = ini_dir_ / path;
  }
  *val = path;

  return OkStatus();
}

}  // namespace llama
