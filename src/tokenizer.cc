#include "tokenizer.h"

#include <stdint.h>
#include <math.h>

namespace llama {

struct LlamaTokenizer::TokenInfo {
  int id;
  float weight;
  std::string token_string;
  int8_t flag;
};

struct LlamaTokenizer::Bigram {
  Symbol *left;
  Symbol *right;
  float weight;
  int merged_token_id;

  bool operator<(Bigram &rhs) const {
    return weight < rhs.weight;
  }
};

// symbol linked list
struct LlamaTokenizer::Symbol {
  Symbol *prev;
  Symbol *next;
  int token_id;

  bool valid() const { return token_id != kInvalidToken; }
};

Status LlamaTokenizer::ReadModel(ReadableFile *fp) {
  std::string s;
  RETURN_IF_ERROR(fp->ReadString(4, &s));
  if (s != "LLsp") {
    RETURN_ABORTED() << "bad format";
  }

  int32_t num_tokens = 0;
  RETURN_IF_ERROR(fp->ReadValue(&num_tokens));

  // read the list of token info
  tokens_.clear();
  for (int token_id = 0; token_id < num_tokens; ++token_id) {
    uint8_t n_bytes;
    TokenInfo info;

    // read the record
    RETURN_IF_ERROR(fp->ReadValue(&info.flag));
    RETURN_IF_ERROR(fp->ReadValue(&n_bytes));
    if (n_bytes) {
      // token string for unknown token, control token and unused token is
      // empty
      RETURN_IF_ERROR(fp->ReadString(n_bytes, &info.token_string));
    }
    RETURN_IF_ERROR(fp->ReadValue(&info.weight));

    info.id = token_id;
    if (info.flag & kUnknown) {
      // handle unknown token
      if (unk_id_ != kInvalidToken) {
        RETURN_ABORTED() << "bad format, too many unknown tokens";
      }
      unk_id_ = token_id;
    }

    tokens_.emplace_back(std::move(info));
  }

  // ensure magic number
  int16_t magic_number = 0;
  RETURN_IF_ERROR(fp->ReadValue(&magic_number));
  if (magic_number != kMagicNumber) {
    RETURN_ABORTED() << "bad format";
  }

  // build token dict
  token_dict_.clear();
  for (const TokenInfo &info : tokens_) {
    if (!info.flag) {
      // flag == 0 means it's a normal token (not control token, unknown token
      // or unused token)
      token_dict_[info.token_string] = &info;
    }
  }

  return OkStatus();
}

Status LlamaTokenizer::CheckModel() {
  // ensure all byte are present in the vocabulary
  std::string tok = " ";
  for (int ch = 0; ch < 256; ++ch) {
    tok[0] = static_cast<char>(ch);
    if (token_dict_.find(tok) == token_dict_.end()) {
      RETURN_ABORTED() << "bad format, byte << " << ch << " not exist in model";
    }
  }
}

std::vector<int> LlamaTokenizer::Encode(const std::string &s) const {
  if (s.empty()) {
    return std::vector<int>();
  }

  std::vector<Symbol> symbols(s.size());

  // convert all bytes in s to symbols linked list
  InitSymbolList(s, MakeSpan(symbols));

  // put all possible two-bytes bigram to queue
  std::priority_queue<Bigram> queue;
  Symbol *begin_sym = &symbols[0],
         *p = begin_sym,
         *q = p->next;
  while (q) {
    AddBigramIfExist(p, q, &queue);
    p = q;
    q = q->next;
  }

  // loop until there is no bigram candidates
  while (!queue.empty()) {
    Bigram bigram = queue.top();
    queue.pop();

    if (bigram.left->valid() && bigram.right->valid()) {
      Symbol *symbol = MergeBigramSymbols(bigram);
      AddBigramIfExist(symbol->prev, symbol, &queue);
      AddBigramIfExist(symbol, symbol->next, &queue);
    }
  }

  // get symbol list
  std::vector<int> token_ids;
  Symbol *p = begin_sym;
  while (p) {
    token_ids.push_back(p->token_id);
    p = p->next;
  }

  return token_ids;
}

void LlamaTokenizer::AddBigramIfExist(
    Symbol *left,
    Symbol *right,
    std::priority_queue<Bigram> *queue) const {
  if (left == nullptr || right == nullptr) {
    return;
  }

  std::string tok;
  tok += token_string(left->token_id);
  tok += token_string(right->token_id);

  auto it = token_dict_.find(tok);
  if (it == token_dict_.end()) {
    return;
  }

  Bigram bigram;
  bigram.left = left;
  bigram.right = right;
  bigram.weight = it->second->weight;
  bigram.merged_token_id = it->second->id;
  queue->push(bigram);
}

LlamaTokenizer::Symbol *LlamaTokenizer::MergeBigramSymbols(
    const Bigram &bigram) const {
  Symbol *left = bigram.left;
  Symbol *right = bigram.right;
  Symbol *next = right->next;
  
  left->token_id = bigram.merged_token_id;
  left->next = next;
  if (next) {
    next->prev = left;
  }
  right->token_id = kInvalidToken;
  right->next = nullptr;
  right->prev = nullptr;

  return left;
}

void LlamaTokenizer::InitSymbolList(const std::string &s,
                                    Span<Symbol> symbols) const {
  LL_CHECK(symbols.size() == s.size());

  Symbol *prev = nullptr;
  std::string tok = " ";
  for (int i = 0; i < s.size(); ++i) {
    char ch = s[i];
    Symbol *symbol = &symbols[i];

    tok[0] = ch;
    int token_id = FindToken(tok);
    LL_CHECK(token_id != unk_id());

    symbol->token_id = token_id;
    symbol->prev = prev;
    symbol->next = nullptr;
    if (prev) {
      prev->next = symbol;
    }

    prev = symbol;
  }
}

}  // namespace llama
