#include "ini_parser.h"

#include "reader.h"
#include "strings.h"
#include "util.h"

namespace llama {

IniParser::IniParser() {}

StatusOr<IniParser> IniParser::Read(const std::string &filename) {
  auto fp = ReadableFile::Open(filename);
  RETURN_IF_ERROR(fp);

  // Read line by line
  std::string line;
  std::string section;
  std::unordered_map<std::string, std::string> table;
  while (IsOK(fp->ReadLine(&line))) {
    // section line: [<section-name>]
    line = strings::Trim(line);
    if (line.front() == '[' && line.back() == ']') {
      section = line.substr(1, line.size() - 2);
      section = strings::Trim(section);
      if (section.empty()) {
        RETURN_ABORTED() << filename << ": section is empty: " << line;
      }
    } else if (line.empty() || line.front() == ';') {
      // empty line and comment line, do nothing
    } else {
      // key-value pair line: key=value
      if (section.empty()) {
        RETURN_ABORTED() << filename
                         << ": key-value pair occured before any section: "
                         << line;
      }

      auto row = strings::Split(line, "=");
      if (row.size() != 2) {
        RETURN_ABORTED() << filename << ": invalid line: " << line;
      }
      std::string key = strings::ToLower(strings::Trim(row[0]));
      std::string value = strings::Trim(row[1]);
      if (value.empty()) {
        RETURN_ABORTED() << filename << ": value is empty, key = " << key;
      }

      table[section + "::" + key] = value;
    }
  }

  std::unique_ptr<IniParser> parser{new IniParser()};
  parser->filename_ = filename;
  parser->table_ = std::move(table);
  return parser;
}

template<>
Status IniParser::Get(const std::string &section,
                      const std::string &key,
                      std::string *val) const {
  std::string s;
  std::string section_key = section + "::" + key;
  auto it = table_.find(section_key);
  if (it == table_.end()) {
    RETURN_ABORTED() << "section and key not found: " << section_key;
  }

  *val = it->second;
  return OkStatus();
}

template<>
Status IniParser::Get(const std::string &section,
                      const std::string &key,
                      int *val) const {
  std::string s;
  RETURN_IF_ERROR(Get(section, key, &s));

  int v;
  RETURN_IF_ERROR(strings::Atoi(s, &v));
  *val = v;

  return OkStatus();
}


template<>
Status IniParser::Get(const std::string &section,
                      const std::string &key,
                      util::Path *val) const {
  std::string s;
  RETURN_IF_ERROR(Get(section, key, &s));

  util::Path path(s);
  if (!path.isabs()) {
    util::Path ini_dir = util::Path(filename_).dirname();
    path = ini_dir / path;
  }
  *val = path;

  return OkStatus();
}

}  // namespace llama
