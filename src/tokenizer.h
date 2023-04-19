#ifndef LLAMA_CC_TOKENIZER_H_
#define LLAMA_CC_TOKENIZER_H_

#include <queue>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "reader.h"
#include "status.h"
#include "util.h"

namespace llama {

// interface for vocabulary class that supports finding token_id by token
// string or getting token string by token id.
class Vocab {
 public:
  ~Vocab() = default;

  // find token id by token string. returns unk_id if the token not exist in
  // the vocabulary.
  virtual int FindToken(const std::string &token) const = 0;

  // get token string by token id. token_id must in range [0, vocab_size),
  // otherwise it will trigger a FATAL error. If the id is unknown token,
  // control token or unused token, an empty string will be returned.
  virtual const std::string &token_string(int token_id) const = 0;

  virtual int vocab_size() const = 0;
  virtual int unk_id() const = 0;
};

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
  static constexpr int kByte = 4;
  static constexpr int kUnused = 8;
  static constexpr int kInvalidToken = -1;
  static constexpr char kUnderscore[] = "\xe2\x96\x81";

  std::unordered_map<std::string, const TokenInfo *> token_dict_;
  std::vector<TokenInfo> tokens_;
  int byte_id_[256];

  int unk_id_;
  int underscore_id_;
  
  LlamaTokenizer();

  // read model from fp
  Status ReadModel(ReadableFile *fp);
  Status ReadMagicNumber(ReadableFile *fp);
  Status ReadTokenInfoRecord(ReadableFile *fp, TokenInfo *info);
  Status InitModel();
  Status CheckModel();
};

}  // namespace llama

#endif  // LLAMA_CC_TOKENIZER_H_
