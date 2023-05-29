#ifndef LLAMA_CC_TOKENIZER_H_
#define LLAMA_CC_TOKENIZER_H_

#include <queue>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "ini_config.h"

namespace llama {

// interface for vocabulary class that supports finding token_id by token
// string or getting token_string abd token_piece by id.
class Vocab {
 public:
  // id for invalid token.
  static constexpr int kInvalidToken = -1;

  virtual ~Vocab() = default;

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

  // total number of tokens in the vocabulary.
  virtual int vocab_size() const = 0;

  // id for unknown token.
  virtual int unk_id() const = 0;
};

// interface for Tokenizer.
class Tokenizer {
 public:
  // create an instance of Tokenizer from config.
  static expected_ptr<Tokenizer> FromConfig(const IniSection &config);
 
  virtual ~Tokenizer() = default;
  
  // encode the input string to a list of token_ids.
  virtual std::vector<int> Encode(const std::string &s) const = 0;

  // return the vocabulary of this tokenizer.
  virtual const Vocab *vocab() const = 0;
};

}  // namespace llama

#endif  // LLAMA_CC_TOKENIZER_H_
