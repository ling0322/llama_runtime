#include "tokenizer.h"

#include <stdint.h>
#include <math.h>
#include "pool.h"

namespace llama {

struct LlamaTokenizer::TokenInfo {
  int id;
  float weight;
  std::string token_string;
  int8_t flag;
};

// ----------------------------------------------------------------------------
// class LlamaTokenizer::Encoder
// ----------------------------------------------------------------------------

// internal implementation of LlamaTokenizer::Encode()
class LlamaTokenizer::Encoder {
 private:
  struct Symbol;
  struct Bigram;

 public:
  Encoder(const LlamaTokenizer *model);

  std::vector<int> Encode(const std::string &s);

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

 private:
  static constexpr int kSymbolPoolBlockSize = 256;

  struct Bigram {
    Symbol *left;
    Symbol *right;
    float weight;
    int merged_token_id;

    bool operator<(const Bigram &rhs) const {
      return weight < rhs.weight;
    }
  };

  // symbol linked list
  struct Symbol {
    Symbol *prev;
    Symbol *next;
    int token_id;

    bool valid() const { return token_id != kInvalidToken; }
  };

  Pool<Symbol, kSymbolPoolBlockSize> symbol_pool_;
  Symbol *header_;
  std::priority_queue<Bigram> queue_;
  const LlamaTokenizer *model_;
};

LlamaTokenizer::Encoder::Encoder(const LlamaTokenizer *model)
    : header_(nullptr),
      model_(model) {}

void LlamaTokenizer::Encoder::InitQueue() {
  Symbol *p = header_->next,
         *q = p->next;
  while (q) {
    AddBigramIfExist(p, q);
    p = q;
    q = q->next;
  }
}

std::vector<int> LlamaTokenizer::Encoder::GetSymbolList() {
  std::vector<int> token_ids;
  Symbol *p = header_->next;
  while (p) {
    token_ids.push_back(p->token_id);
    p = p->next;
  }

  return token_ids;
}

std::vector<int> LlamaTokenizer::Encoder::Encode(const std::string &s) {
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


void LlamaTokenizer::Encoder::AddBigramIfExist(Symbol *left, Symbol *right) {
  if (left == header_ || right == nullptr) {
    return;
  }

  std::string tok;
  tok += model_->token_string(left->token_id);
  tok += model_->token_string(right->token_id);

  auto it = model_->token_dict_.find(tok);
  if (it == model_->token_dict_.end()) {
    return;
  }

  Bigram bigram;
  bigram.left = left;
  bigram.right = right;
  bigram.weight = it->second->weight;
  bigram.merged_token_id = it->second->id;
  queue_.push(bigram);
}

LlamaTokenizer::Encoder::Symbol *LlamaTokenizer::Encoder::Merge(
    const Bigram &bigram) {
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

  right->token_id = kInvalidToken;
  right->next = nullptr;
  right->prev = nullptr;

  left->token_id = kInvalidToken;
  left->next = nullptr;
  left->prev = nullptr;

  return merged;
}

void LlamaTokenizer::Encoder::InitSymbolList(const std::string &s) {
  Symbol *header = symbol_pool_.Alloc();
  Symbol *prev = header;

  std::string tok = " ";
  for (int i = 0; i < s.size(); ++i) {
    char ch = s[i];
    Symbol *symbol = symbol_pool_.Alloc();

    tok[0] = ch;
    int token_id = model_->FindToken(tok);
    CHECK(token_id != model_->unk_id());

    symbol->token_id = token_id;
    symbol->prev = prev;
    symbol->next = nullptr;

    prev->next = symbol;
    prev = symbol;
  }

  header_ = header;
}

// ----------------------------------------------------------------------------
// class LlamaTokenizer
// ----------------------------------------------------------------------------

LlamaTokenizer::LlamaTokenizer(): unk_id_(kInvalidToken) {}

StatusOr<LlamaTokenizer> LlamaTokenizer::FromModel(
    const std::string &filename) {
  std::unique_ptr<LlamaTokenizer> tokenizer(new LlamaTokenizer());
  auto fp = ReadableFile::Open(filename);
  RETURN_IF_ERROR(fp) << "create LlamaTokenizer failed";

  RETURN_IF_ERROR(tokenizer->ReadModel(fp.get()));
  return tokenizer;
}

Status LlamaTokenizer::ReadModel(ReadableFile *fp) {
  std::string s;
  RETURN_IF_ERROR(fp->ReadString(4, &s));
  if (s != "LLsp") {
    RETURN_ABORTED() << "bad format";
  }

  int32_t num_tokens = 0;
  RETURN_IF_ERROR(fp->ReadValue(&num_tokens));

  // ensure magic number
  int16_t magic_number = 0;
  RETURN_IF_ERROR(fp->ReadValue(&magic_number));
  if (magic_number != kMagicNumber) {
    RETURN_ABORTED() << "bad format";
  }


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
      LOG(INFO) << "unk_id " << token_id;
      // handle unknown token
      if (unk_id_ != kInvalidToken) {
        RETURN_ABORTED() << "bad format, too many unknown tokens";
      }
      unk_id_ = token_id;
    }

    tokens_.emplace_back(std::move(info));
  }

  // ensure magic number
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

  return OkStatus();
}

std::vector<int> LlamaTokenizer::Encode(const std::string &s) const {
  return Encoder(this).Encode(s);
}

std::vector<std::string> LlamaTokenizer::EncodeAsPieces(
    const std::string &s) const {
  std::vector<std::string> tokens;

  std::vector<int> token_ids = Encode(s);
  for (int token_id : token_ids) {
    tokens.emplace_back(token_string(token_id));
  }

  return tokens;
}

const std::string &LlamaTokenizer::token_string(int token_id) const {
  CHECK(token_id >= 0 && token_id < vocab_size());

  return tokens_[token_id].token_string;
}

int LlamaTokenizer::FindToken(const std::string &token) const {
  auto it = token_dict_.find(token);
  if (it == token_dict_.end()) {
    return unk_id();
  }

  return it->second->id;
}

int LlamaTokenizer::vocab_size() const {
  return tokens_.size();
}

int LlamaTokenizer::unk_id() const {
  return unk_id_;
}

}  // namespace llama
