#ifndef LLMRUNTIME_VOCAB_H_
#define LLMRUNTIME_VOCAB_H_

#include <queue>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace llama {

// interface for vocabulary class that supports finding token_id by token
// string or getting token string by token id.
class Vocab {
 public:
  // id for invalid token.
  static constexpr int kInvalidToken = -1;

  ~Vocab() = default;

  // find token id by token string or the byte ord. returns unk_id if the token
  // not exist in the vocabulary.
  virtual int FindToken(const std::string &token) const = 0;

  // get token string by token id. token_id must in range [0, vocab_size),
  // otherwise it will trigger a FATAL error. If the id is unknown token,
  // control token or unused token, an empty string will be returned.
  virtual const std::string &token_string(int token_id) const = 0;

  virtual int vocab_size() const = 0;
  virtual int unk_id() const = 0;
};

}  // namespace llama

#endif  // LLMRUNTIME_VOCAB_H_
