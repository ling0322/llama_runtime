#include "ini_config.h"

#include <tuple>
#include "reader.h"
#include "strings.h"
#include "util.h"

namespace llama {

// -- class IniConfig ----------------------------------------------------------

IniConfig::IniConfig() {}

std::unique_ptr<IniConfig> IniConfig::read(const std::string &filename) {
  std::unique_ptr<IniConfig> config{new IniConfig()};

  auto fp = ReadableFile::open(filename);
  Scanner scanner{fp.get()};

  enum State {
    kBegin,
    kSelfLoop,
  };

  util::Path ini_dir = util::Path(filename).dirname();
  IniSection section{ini_dir};
  State state = kBegin;
  while (scanner.scan()) {
    std::string line = strings::Trim(scanner.getText());
    if (state == kBegin) {
      if (isEmptyLine(line)) {
        // self-loop
      } else if (isHeader(line)) {
        section._name = parseHeader(line);
        state = kSelfLoop;
      } else {
        throw AbortedException(fmt::sprintf("invalid line: %s", line));
      }
    } else if (state == kSelfLoop) {
      if (isEmptyLine(line)) {
        // do nothing.
      } else if (isHeader(line)) {
        config->_table.emplace(section.getName(), std::move(section));
        section = IniSection{ini_dir};
        section._name = parseHeader(line);
      } else {
        std::string key, value;
        std::tie(key, value) = parseKeyValue(line);
        section._kvTable[key] = value;
      }
    } else {
      NOT_IMPL();
    }
  }

  if (state == kBegin) {
    throw AbortedException("ini file is empty.");
  }

  config->_table.emplace(section.getName(), std::move(section));

  return config;
}

bool IniConfig::hasSection(const std::string &section) const {
  auto it = _table.find(section);
  return it != _table.end();
}


const IniSection &IniConfig::getSection(const std::string &name) const {
  auto it = _table.find(name);
  CHECK(it != _table.end()) << "section not found: " << name;

  return it->second;
}

bool IniConfig::isEmptyLine(const std::string &s) {
  if (s.empty() || s.front() == ';') {
    return true;
  } else {
    return false;
  }
}

bool IniConfig::isHeader(const std::string &s) {
  if (s.front() == '[' && s.back() == ']') {
    return true;
  } else {
    return false;
  }
}

std::string IniConfig::parseHeader(const std::string &s) {
  if (!isHeader(s)) {
    throw AbortedException(fmt::sprintf("invalid line: %s", s));
  }
  
  std::string name = s.substr(1, s.size() - 2);
  name = strings::Trim(name);
  if (name.empty()) {
    throw AbortedException(fmt::sprintf("invalid ini section: %s", s));
  }

  return name;
}

std::pair<std::string, std::string> IniConfig::parseKeyValue(const std::string &s) {
  auto row = strings::Split(s, "=");
  if (row.size() != 2) {
    throw AbortedException(fmt::sprintf("invalid line: %s", s));
  }
  std::string key = strings::ToLower(strings::Trim(row[0]));
  std::string value = strings::Trim(row[1]);
  if (key.empty() || value.empty()) {
    throw AbortedException(fmt::sprintf("invalid line: %s", s));
  }

  return std::make_pair(key, value);
}


// -- class IniSection ---------------------------------------------------------

IniSection::IniSection(const util::Path &iniDir) : _iniDir(iniDir) {}

std::string IniSection::getString(const std::string &key) const {
  auto it = _kvTable.find(key);
  if (it == _kvTable.end()) {
    throw AbortedException(fmt::sprintf("key not found (ini_session=%s): %s", _name, key));
  }

  return it->second;
}

int IniSection::getInt(const std::string &key) const {
  std::string s = getString(key);
  return strings::Atoi(s);
}

bool IniSection::getBool(const std::string &key) const {
  std::string s = getString(key);

  s = strings::ToLower(s);
  if (s == "true" || s == "1") {
    return true;
  } else if (s == "false" || s == "0") {
    return false;
  } else {
    throw AbortedException(fmt::sprintf("invalid bool value: %s", s));
  }

  // never reach here.
  NOT_IMPL();
  return false;
}

util::Path IniSection::getPath(const std::string &key) const {
  std::string s = getString(key);

  util::Path path(s);
  if (!path.isabs()) {
    return _iniDir / path;
  } else {
    return path;
  }
}

}  // namespace llama
