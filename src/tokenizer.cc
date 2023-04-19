#include "tokenizer.h"

#include <stdint.h>
#include <math.h>

#include "strings.h"

namespace llama {

struct LlamaTokenizer::TokenInfo {
  int id;
  float weight;
  std::string token_string;
  int token_byte;
  int8_t flag;

  constexpr bool is_special_token() const { return flag != 0; }
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

  util::Pool<Symbol, kSymbolPoolBlockSize> symbol_pool_;
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
  if (left == header_ || right == nullptr ||
      model_->tokens_[left->token_id].is_special_token() ||
      model_->tokens_[right->token_id].is_special_token()) {
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
  // preprocess sentence
  Symbol *header = symbol_pool_.Alloc();
  header->prev = nullptr;
  header->token_id = kInvalidToken;

  // prefix (_)
  Symbol *prefix = symbol_pool_.Alloc();
  prefix->prev = header;
  prefix->token_id = model_->underscore_id_;
  prefix->next = nullptr;
  header->next = prefix;

  Symbol *prev = prefix;
  std::vector<std::string> utf8_chars = strings::SplitToUtf8Chars(s);
  for (const std::string &utf8_ch : utf8_chars) {
    int token_id = utf8_ch == " " ? model_->underscore_id_
                                  : model_->FindToken(utf8_ch);
    if (token_id == model_->unk_id()) {
      // symbol not found in the vocab, split it into bytes
      for (char ch : utf8_ch) {
        Symbol *symbol = symbol_pool_.Alloc();

        symbol->token_id = model_->byte_id_[static_cast<uint8_t>(ch)];
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

// ---------------------------------------------------------------------------+
// class LlamaTokenizer                                                       |
// ---------------------------------------------------------------------------+

LlamaTokenizer::LlamaTokenizer(): unk_id_(kInvalidToken) {
  for (int i = 0; i < 256; ++i) {
    byte_id_[i] = kInvalidToken;
  }
}

StatusOr<LlamaTokenizer> LlamaTokenizer::FromModel(
    const std::string &filename) {
  std::unique_ptr<LlamaTokenizer> tokenizer(new LlamaTokenizer());
  auto fp = ReadableFile::Open(filename);
  RETURN_IF_ERROR(fp) << "create LlamaTokenizer failed";

  RETURN_IF_ERROR(tokenizer->ReadModel(fp.get()));
  RETURN_IF_ERROR(tokenizer->CheckModel());
  return tokenizer;
}

Status LlamaTokenizer::ReadModel(ReadableFile *fp) {
  std::string s;
  RETURN_IF_ERROR(fp->ReadString(4, &s));
  if (s != "LLsp") {
    RETURN_ABORTED() << "bad format (header)";
  }

  int32_t num_tokens = 0;
  RETURN_IF_ERROR(fp->ReadValue(&num_tokens));
  RETURN_IF_ERROR(ReadMagicNumber(fp));

  // read the list of token info
  tokens_.clear();
  for (int token_id = 0; token_id < num_tokens; ++token_id) {
    TokenInfo info;
    RETURN_IF_ERROR(ReadTokenInfoRecord(fp, &info));
    info.id = token_id;

    tokens_.emplace_back(std::move(info));
  }

  RETURN_IF_ERROR(ReadMagicNumber(fp));
  RETURN_IF_ERROR(InitModel());
  return OkStatus();
}

Status LlamaTokenizer::ReadMagicNumber(ReadableFile *fp) {
  // ensure magic number
  int16_t magic_number = 0;
  RETURN_IF_ERROR(fp->ReadValue(&magic_number));
  if (magic_number != kMagicNumber) {
    RETURN_ABORTED() << "bad format (magic number)";
  }

  return OkStatus();
}

Status LlamaTokenizer::InitModel() {
  // build token_dict_ and byte_id_
  token_dict_.clear();
  for (const TokenInfo &info : tokens_) {
    if (!info.flag) {
      // flag == 0 means it's a normal token (not control token, unknown token
      // or unused token)
      token_dict_[info.token_string] = &info;
    } else if (info.flag & kByte) {
      // single byte token
      byte_id_[info.token_byte] = info.id;
    } else if (info.flag & kUnknown) {
      if (unk_id_ != kInvalidToken) {
        RETURN_ABORTED() << "bad format (too many unknown tokens)";
      }
      unk_id_ = info.id;
    }
  }

  // find id for underscore character
  auto it_underscore = token_dict_.find(kUnderscore);
  if (it_underscore == token_dict_.end()) {
    RETURN_ABORTED() << "bad format (no symbol for underscore)";
  }
  underscore_id_ = it_underscore->second->id;

  return OkStatus();
}

Status LlamaTokenizer::ReadTokenInfoRecord(ReadableFile *fp, TokenInfo *info) {
  uint8_t n_bytes;

  RETURN_IF_ERROR(fp->ReadValue(&info->flag));
  RETURN_IF_ERROR(fp->ReadValue(&n_bytes));
  std::string token_string;
  if (n_bytes) {
    RETURN_IF_ERROR(fp->ReadString(n_bytes, &token_string));
  }
  if (!info->flag) {
    info->token_string = std::move(token_string);
    info->token_byte = kInvalidToken;
  } else if (info->flag & kByte) {
    if (token_string.size() != 1) {
      RETURN_ABORTED() << "bad format (byte)";
    }
    info->token_byte = static_cast<uint8_t>(token_string[0]);
    info->token_string = strings::Sprintf("<0x%02X>", info->token_byte);
  }
  RETURN_IF_ERROR(fp->ReadValue(&info->weight));

  return OkStatus();
}

Status LlamaTokenizer::CheckModel() {
  for (int ch = 0; ch < 256; ++ch) {
    if (byte_id_[ch] == kInvalidToken) {
      RETURN_ABORTED() << "bad format, byte << " << ch << " not exist in model";
    }
  }

  if (unk_id_ == kInvalidToken) {
    RETURN_ABORTED() << "bad model (no unknown token)";
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
