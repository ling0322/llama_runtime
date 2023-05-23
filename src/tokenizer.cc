#include "tokenizer.h"

#include <stdint.h>
#include <math.h>
#include <functional>

#include "bpe_encoder.h"

namespace llama {

BpeTokenizer::BpeTokenizer(std::unique_ptr<BpeModel> &&model)
    : model_(std::move(model)) {}

std::vector<int> BpeTokenizer::Encode(const std::string &s) const {
  BpeEncoder encoder(model_.get());
  return encoder.Encode(s);
}

std::vector<std::string> BpeTokenizer::EncodeAsPieces(
    const std::string &s) const {
  std::vector<std::string> tokens;
  std::vector<int> token_ids = Encode(s);
  for (int token_id : token_ids) {
    tokens.emplace_back(model_->token_string(token_id));
  }
  return tokens;
}

}  // namespace llama
