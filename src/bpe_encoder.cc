#include "bpe_encoder.h"

#include <stdint.h>
#include <math.h>
#include <functional>

#include "strings.h"

namespace llama {

// -- class BpeEncoder --------------------------------------------------------

BpeEncoder::BpeEncoder(const BpeModel *model)
    : model_(model),
      header_(nullptr) {}

void BpeEncoder::InitQueue() {
  Symbol *p = header_->next,
         *q = p->next;
  while (q) {
    AddBigramIfExist(p, q);
    p = q;
    q = q->next;
  }
}

std::vector<int> BpeEncoder::GetSymbolList() {
  std::vector<int> token_ids;
  Symbol *p = header_->next;
  while (p) {
    token_ids.push_back(p->token_id);
    p = p->next;
  }

  return token_ids;
}

std::vector<int> BpeEncoder::Encode(const std::string &s) {
  InitSymbolList(s);
  InitQueue();

  // loop until there is no bigram candidates
  while (!queue_.empty()) {
    Bigram bigram = queue_.top();
    queue_.pop();

    if (bigram.left->valid() && bigram.right->valid()) {
      Symbol *symbol = Merge(bigram);
      AddBigramIfExist(symbol->prev, symbol);
      AddBigramIfExist(symbol, symbol->next);
    }
  }

  return GetSymbolList();
}

void BpeEncoder::AddBigramIfExist(Symbol *left, Symbol *right) {
  if (left == header_ || right == nullptr ||
      model_->is_special(right->token_id) ||
      model_->is_special(left->token_id)) {
    return;
  }

  int left_id = left->token_id;
  int right_id = right->token_id;

  int merged_token_id;
  float cost;
  std::tie(merged_token_id, cost) = model_->FindMerge(left_id, right_id);
  if (merged_token_id == Vocab::kInvalidToken) {
    return;
  }

  Bigram bigram;
  bigram.left = left;
  bigram.right = right;
  bigram.cost = cost;
  bigram.merged_token_id = merged_token_id;
  queue_.push(bigram);
}

BpeEncoder::Symbol *BpeEncoder::Merge(const Bigram &bigram) {
  Symbol *left = bigram.left;
  Symbol *right = bigram.right;
  Symbol *next = right->next;
  Symbol *prev = left->prev;

  Symbol *merged = symbol_pool_.Alloc();
  merged->token_id = bigram.merged_token_id;
  merged->next = next;
  merged->prev = prev;
  if (next) {
    next->prev = merged;
  }
  // prev do not need to check since there is a header node
  prev->next = merged;

  right->token_id = Vocab::kInvalidToken;
  right->next = nullptr;
  right->prev = nullptr;

  left->token_id = Vocab::kInvalidToken;
  left->next = nullptr;
  left->prev = nullptr;

  return merged;
}

void BpeEncoder::InitSymbolList(const std::string &s) {
  // preprocess sentence
  Symbol *header = symbol_pool_.Alloc();
  header->prev = nullptr;
  header->token_id = Vocab::kInvalidToken;

  // prefix (_)
  Symbol *prefix = symbol_pool_.Alloc();
  prefix->prev = header;
  prefix->token_id = model_->space_id();
  prefix->next = nullptr;
  header->next = prefix;

  Symbol *prev = prefix;
  std::vector<std::string> utf8_chars = strings::SplitToUtf8Chars(s);
  for (const std::string &utf8_ch : utf8_chars) {
    int token_id = utf8_ch == " " ? model_->space_id()
                                  : model_->FindToken(utf8_ch);
    if (token_id == model_->unk_id()) {
      // symbol not found in the vocab, split it into bytes
      for (char ch : utf8_ch) {
        Symbol *symbol = symbol_pool_.Alloc();

        symbol->token_id = model_->byte_id(static_cast<uint8_t>(ch));
        symbol->prev = prev;
        symbol->next = nullptr;

        prev->next = symbol;
        prev = symbol;
      }
    } else {
      Symbol *symbol = symbol_pool_.Alloc();

      symbol->token_id = token_id;
      symbol->prev = prev;
      symbol->next = nullptr;

      prev->next = symbol;
      prev = symbol;
    }
  }

  header_ = header;
}

}  // namespace llama
