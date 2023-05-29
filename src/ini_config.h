// Created at 2017-4-19

#ifndef LLMRUNTIME_INI_PARSER_H_
#define LLMRUNTIME_INI_PARSER_H_

#include <map>
#include <string>
#include <unordered_map>
#include "reader.h"
#include "status.h"
#include "util.h"

namespace llama {

class IniSection;

// IniConfig is a class to read ini configuration file.
class IniConfig {
 public:
  // Read configuration from filename.
  static expected_ptr<IniConfig> Read(const std::string &filename);

  // get section by name. CHECK() will fail if section not found.
  const IniSection &section(const std::string &name) const;

  // returns true if section presents in the ini config.
  bool has_section(const std::string &section) const;

  // ensure section exists. Returns OutOfRangeError if not found.
  Status EnsureSection(const std::string &name) const;

  // Get filename.
  const std::string &filename() const { return filename_; }

 private:  
  std::string filename_;

  // map (section, key) -> value
  std::map<std::string, IniSection> table_;
  
  IniConfig();

  static bool IsEmptyLine(const std::string &s);
  static bool IsHeader(const std::string &s);
  static Status ParseHeader(const std::string &s, std::string *name);
  static Status ParseKeyValue(const std::string &s,
                              std::string *key,
                              std::string *value);
};

// one section in ini config.
class IniSection {
 public:
  friend class IniConfig;

  // get a value by section and key. return OutOfRangeError() if the section
  // and key not exist. return Abort() if the type mismatch.
  // supported types: 
  //   - int
  //   - string
  //   - bool
  //   - util::Path (relative to this ini file)
  template<typename T>
  Status Get(const std::string &key, T *val) const;

  // returns true if the key exists.
  bool has_key(const std::string &key);

  // name of the section.
  const std::string &name() const { return name_; }

 private:
  std::unordered_map<std::string, std::string> kv_table_;
  std::string name_;
  util::Path ini_dir_;

  IniSection(const util::Path &ini_dir);
};

}  // namespace llama

#endif  // LLMRUNTIME_INI_PARSER_H_
