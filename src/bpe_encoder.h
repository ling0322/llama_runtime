#ifndef LLMRUNTIME_BPE_ENCODER_H_
#define LLMRUNTIME_BPE_ENCODER_H_

#include <queue>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "bpe_model.h"
#include "vocab.h"


namespace llama {

// String to BPE token-ids encoder.
class BpeEncoder : private util::NonCopyable {
 public:
  BpeEncoder(const BpeModel *model);

  // encode string to token ids. 
  std::vector<int> Encode(const std::string &s);

 private:
  static constexpr int kSymbolPoolBlockSize = 256;

  // symbol linked list
  struct Symbol {
    Symbol *prev;
    Symbol *next;
    int token_id;

    bool valid() const { return token_id != Vocab::kInvalidToken; }
  };

  struct Bigram {
    Symbol *left;
    Symbol *right;
    float cost;
    int merged_token_id;

    bool operator>(const Bigram &rhs) const {
      return cost > rhs.cost;
    }
  };

  const BpeModel *model_;
  util::Pool<Symbol, kSymbolPoolBlockSize> symbol_pool_;
  Symbol *header_;
  std::priority_queue<Bigram, std::vector<Bigram>, std::greater<Bigram>> queue_;

  // initialize the symbol linked list from string `s` and store the pointer of
  // header node to `header_`.
  void InitSymbolList(const std::string &s);

  // initialize the queue by putting all possible two-bytes bigram to queue
  void InitQueue();

  // add bigram (left, right) to queue if token left+right exists
  void AddBigramIfExist(Symbol *left, Symbol *right);

  // merge bigram (left, right) into one symbol, then clear original left and
  // right symbols and return pointer to the merged one.
  Symbol *Merge(const Bigram &bigram);

  // get the final symbol list from linked list pointered by header_
  std::vector<int> GetSymbolList();
};


}  // namespace llama

#endif  // LLMRUNTIME_BPE_ENCODER_H_
