#ifndef LLAMA_CC_TOKENIZER_H_
#define LLAMA_CC_TOKENIZER_H_

#include <queue>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "ini_parser.h"
#include "bpe_model.h"


namespace llama {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;
  
  virtual std::vector<int> Encode(const std::string &s) const = 0;
  virtual std::vector<std::string> EncodeAsPieces(
      const std::string &s) const = 0;
};

class BpeTokenizer : public Tokenizer {
 public:
  BpeTokenizer(std::unique_ptr<BpeModel> &&model);

  // implement interface Tokenizer
  std::vector<int> Encode(const std::string &s) const override;
  std::vector<std::string> EncodeAsPieces(const std::string &s) const override;

 private:
  std::unique_ptr<BpeModel> model_;
};

}  // namespace llama

#endif  // LLAMA_CC_TOKENIZER_H_
