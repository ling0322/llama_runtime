#ifndef LLAMA_CC_VOCAB_H_
#define LLAMA_CC_VOCAB_H_

#include "common.h"

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

  // number of tokens in this vocabulary including unknown token, control
  // token and unused token.
  virtual int vocab_size() const = 0;

  // id for unknown token. There should be only one unknown token in the
  // vocabulary.
  virtual int unk_id() const = 0;
};

}  // namespace llama

#endif  // LLAMA_CC_VOCAB_H_
