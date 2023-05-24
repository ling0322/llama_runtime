#ifndef LLAMA_CC_TOKENIZER_H_
#define LLAMA_CC_TOKENIZER_H_

#include <queue>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "ini_parser.h"

namespace llama {

// interface for vocabulary class that supports finding token_id by token
// string or getting token_string abd token_piece by id.
class Vocab {
 public:
  // id for invalid token.
  static constexpr int kInvalidToken = -1;

  ~Vocab() = default;

  // find token id by token string or the byte ord. returns unk_id if the token
  // not exist in the vocabulary.
  virtual int FindToken(const std::string &piece) const = 0;

  // get token bytes by token id. The token id should fall within the range of
  // (0, vocab_size).
  // If the id is unknown token, control token or unused token, an empty string
  // will be returned. To get the display form of a token, please use
  // token_string
  virtual const std::string &token_piece(int token_id) const = 0;

  // get human readable token representation. The token id should fall within
  // the range of (0, vocab_size).
  virtual const std::string &token_string(int token_id) const = 0;

  virtual int vocab_size() const = 0;
  virtual int unk_id() const = 0;
};

// interface for Tokenizer.
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;
  
  // encode the input string to a list of token_ids.
  virtual std::vector<int> Encode(const std::string &s) const = 0;

  // return the vocabulary of this tokenizer.
  virtual const Vocab *vocab() const = 0;
};

// config for BPE tokenizer.
struct BpeConfig {
  // path of the BPE model.
  std::string model_file;

  // true if add a space into the begining of text.
  bool add_prefix_space;

  // true if split by unicode characters before merging. false if split by byte.
  bool split_by_unicode;

  // contructor for the default config.
  BpeConfig();
};

class BpeModel;

// BPE tokenizer.
class BpeTokenizer : public Tokenizer {
 public:
  static StatusOr<BpeTokenizer> FromConfig(const BpeConfig &config);

  // implement interface Tokenizer
  std::vector<int> Encode(const std::string &s) const override;
  const Vocab *vocab() const override;

 private:
  std::unique_ptr<BpeModel> model_;
  BpeConfig config_;

  BpeTokenizer();
};

}  // namespace llama

#endif  // LLAMA_CC_TOKENIZER_H_
