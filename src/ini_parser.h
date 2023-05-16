// Created at 2017-4-19

#ifndef LLMRUNTIME_INI_PARSER_H_
#define LLMRUNTIME_INI_PARSER_H_

#include <string>
#include <unordered_map>
#include "status.h"
#include "util.h"

namespace llama {

// IniParser is a class to read ini configuration file.
class IniParser {
 public:
  // Read configuration from filename.
  static StatusOr<IniParser> Read(const std::string &filename);

  // get a value by section and key. return OutOfRangeError() if the section
  // and key not exist. return Abort() if the type mismatch.
  // supported types: 
  //   - int
  //   - string
  //   - util::Path (relative to this ini file)
  template<typename T>
  Status Get(const std::string &section, const std::string &key, T *val) const;

  // returns true if the section and key exists.
  bool exists(const std::string &section, const std::string &key);

  // Get filename.
  const std::string &filename() const;

 private:  
  std::string filename_;
  std::unordered_map<std::string, std::string> table_;

  IniParser();
};

}  // namespace llama

#endif  // LLMRUNTIME_INI_PARSER_H_
