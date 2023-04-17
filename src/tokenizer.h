#ifndef LLAMA_CC_TOKENIZER_H_
#define LLAMA_CC_TOKENIZER_H_

#include <queue>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "pool.h"
#include "readable_file.h"
#include "status.h"
#include "vocab.h"
#include "util.h"

namespace llama {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;
  
  virtual std::vector<int> Encode(const std::string &s) const = 0;
  virtual std::vector<std::string> EncodeAsPieces(
      const std::string &s) const = 0;
};

// tokenizer for llama model
class LlamaTokenizer : public Tokenizer,
                       public Vocab,
                       private util::NonCopyable {
 public:
  static StatusOr<LlamaTokenizer> FromModel(const std::string &filename);

  // implement interface ITokenizer
  std::vector<int> Encode(const std::string &s) const override;
  std::vector<std::string> EncodeAsPieces(
      const std::string &s) const override;

  // implement interface Vocab
  int FindToken(const std::string &token) const override;
  const std::string &token_string(int token_id) const override;
  int vocab_size() const override;
  int unk_id() const override;

 private:
  class Encoder;
  struct TokenInfo;

  static constexpr int16_t kMagicNumber = 0x55aa;
  static constexpr int kUnknown = 1;
  static constexpr int kControl = 2;
  static constexpr int kUnused = 4;
  static constexpr int kInvalidToken = -1;

  std::unordered_map<std::string, const TokenInfo *> token_dict_;
  std::vector<TokenInfo> tokens_;
  int unk_id_;
  
  LlamaTokenizer();

  // check the tokenizer model and return a fail state with detailed message
  // if the model is invalid.
  Status CheckModel();
  
  // read model from fp
  Status ReadModel(ReadableFile *fp);
};

}  // namespace llama

#endif  // LLAMA_CC_TOKENIZER_H_
