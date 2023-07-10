// Created at 2017-4-19

#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include "llyn/path.h"
#include "llyn/reader.h"

namespace ly {

class IniSection;

// IniConfig is a class to read ini configuration file.
class IniConfig {
 public:
  // Read configuration from filename.
  static std::unique_ptr<IniConfig> read(const std::string &filename);

  // get section by name.
  const IniSection &getSection(const std::string &name) const;

  // returns true if section presents in the ini config.
  bool hasSection(const std::string &section) const;

  // Get filename.
  const std::string &getFilename() const { return _filename; }

 private:  
  std::string _filename;

  // map (section, key) -> value
  std::map<std::string, IniSection> _table;
  
  IniConfig();

  static bool isEmptyLine(const std::string &s);
  static bool isHeader(const std::string &s);
  static std::string parseHeader(const std::string &s);
  static std::pair<std::string, std::string> parseKeyValue(const std::string &s);
};

// one section in ini config.
class IniSection {
 public:
  friend class IniConfig;

  // get a value by section and key.
  std::string getString(const std::string &key) const;
  int getInt(const std::string &key) const;
  bool getBool(const std::string &key) const;
  Path getPath(const std::string &key) const;

  // returns true if the key exists.
  bool hasKey(const std::string &key);

  // name of the section.
  const std::string &getName() const { return _name; }

 private:
  std::unordered_map<std::string, std::string> _kvTable;
  std::string _name;
  Path _iniDir;

  IniSection(const Path &iniDir);
};

}  // namespace ly
