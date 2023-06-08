#include "tokenizer.h"

#include <stdint.h>
#include <math.h>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "ini_config.h"
#include "reader.h"
#include "strings.h"

namespace llama {

// config for BPE tokenizer.
struct BpeConfig {
  // path of the BPE model.
  std::string model_file;

  // true if add a space into the begining of text.
  bool add_prefix_space;

  // true if split by unicode characters before merging. false if split by byte.
  bool split_by_unicode;

  // create the BPE config from ini config.
  static expected_ptr<BpeConfig> FromConfig(const IniSection &config);

  // contructor for the default config.
  BpeConfig();
};

// model for BPE tokenizer.
class BpeModel;

// BPE tokenizer.
class BpeTokenizer : public Tokenizer {
 public:
  static expected_ptr<BpeTokenizer> FromConfig(const BpeConfig &config);

  // implement interface Tokenizer
  std::vector<int> Encode(const std::string &s) const override;
  const Vocab *vocab() const override;

 private:
  std::unique_ptr<BpeModel> model_;
  BpeConfig config_;

  BpeTokenizer();
};

// -- BpeConfig ----------------------------------------------------------------

BpeConfig::BpeConfig()
    : add_prefix_space(true),
      split_by_unicode(true) {}

expected_ptr<BpeConfig> BpeConfig::FromConfig(const IniSection &ini_section) {
  std::unique_ptr<BpeConfig> config = std::make_unique<BpeConfig>();

  bool add_prefix_space = false;
  bool split_by_unicode = false;
  util::Path model_file;

  add_prefix_space = ini_section.getBool("add_prefix_space");
  split_by_unicode = ini_section.getBool("split_by_unicode");
  model_file = ini_section.getPath("model_file");

  config->add_prefix_space = add_prefix_space;
  config->split_by_unicode = split_by_unicode;
  config->model_file = model_file.string();

  return config;
}

// -- BpeModel -----------------------------------------------------------------

// Store tne data from sentence-piece BPE model.
class BpeModel : public Vocab,
                 private util::NonCopyable {
 public:
  // token flags.
  static constexpr int kUnknown = 1;
  static constexpr int kControl = 2;
  static constexpr int kByte = 4;
  static constexpr int kUnused = 8;

  // Read the BpeModel from file. It could not read a SPM model directly,
  // instead, we need to convert the SPM model using tokenizer_exporter.py.
  static expected_ptr<BpeModel> FromModel(const std::string &filename);

  // implement interface Vocab
  int FindToken(const std::string &token) const override;
  const std::string &token_piece(int token_id) const override;
  const std::string &token_string(int token_id) const override;
  int vocab_size() const override;
  int unk_id() const override;

  // given the left and right token-ids, find the merged token-id and cost,
  // return as (id, cost). If no merge record found, return
  // (Vocab::kInvalidToken, +inf)
  std::pair<int, float> FindMerge(int left, int right) const;

  // token-id for space token.
  int space_id() const;

  // return true if the token is either Unknown, Control, SingleByte or Unsed.
  bool is_special(int token_id) const;

  // returns flag of a token.
  int32_t flag(int token_id) const;

  // get token-id for a single byte.
  int byte_id(int byte) const;
  bool byte_token_available() const { return byte_token_available_; }

 private:
  struct TokenInfo;

  static constexpr int16_t kMagicNumber = 0x55aa;

  std::unordered_map<std::string, const TokenInfo *> token_dict_;
  std::vector<TokenInfo> tokens_;

  // standalone tokens for byte.
  std::array<int, 256> byte_id_;
  bool byte_token_available_;

  int unk_id_;
  int space_id_;
  
  BpeModel();

  // read model from fp
  Status ReadModel(ReadableFile *fp);
  Status ReadMagicNumber(ReadableFile *fp);
  Status ReadRecord(ReadableFile *fp, TokenInfo *info);
  Status InitModel();
  Status CheckModel();
};

struct BpeModel::TokenInfo {
  int id;
  float weight;
  std::string token_piece;
  std::string token_string;
  int8_t flag;

  constexpr bool is_special_token() const { return flag != 0; }
};


BpeModel::BpeModel()
    : unk_id_(kInvalidToken),
      byte_token_available_(false) {
  std::fill(byte_id_.begin(), byte_id_.end(), kInvalidToken);
}

expected_ptr<BpeModel> BpeModel::FromModel(const std::string &filename) {
  std::unique_ptr<BpeModel> model(new BpeModel());
  auto fp = ReadableFile::open(filename);

  RETURN_IF_ERROR(model->ReadModel(fp.get()));
  RETURN_IF_ERROR(model->CheckModel());
  return model;
}

Status BpeModel::ReadModel(ReadableFile *fp) {
  std::string s = fp->readString(4);
  if (s != "LLsp") {
    RETURN_ABORTED() << "bad format (header)";
  }

  int32_t num_tokens = fp->readValue<int32_t>();
  RETURN_IF_ERROR(ReadMagicNumber(fp));

  // read the list of token info
  tokens_.clear();
  for (int token_id = 0; token_id < num_tokens; ++token_id) {
    TokenInfo info;
    RETURN_IF_ERROR(ReadRecord(fp, &info));
    info.id = token_id;

    tokens_.emplace_back(std::move(info));
  }

  RETURN_IF_ERROR(ReadMagicNumber(fp));
  RETURN_IF_ERROR(InitModel());
  return OkStatus();
}

Status BpeModel::ReadMagicNumber(ReadableFile *fp) {
  // ensure magic number
  int16_t magic_number = fp->readValue<int16_t>();
  if (magic_number != kMagicNumber) {
    RETURN_ABORTED() << "bad format (magic number)";
  }

  return OkStatus();
}

Status BpeModel::InitModel() {
  // build token_dict_ and byte_id_
  token_dict_.clear();
  

  for (const TokenInfo &info : tokens_) {
    if (!info.flag) {
      // flag == 0 means it's a normal token (not control token, unknown token
      // or unused token)
      token_dict_[info.token_piece] = &info;
    } else if (info.flag & kByte) {
      // single byte token
      CHECK(info.token_piece.size() == 1);
      byte_token_available_ = true;
      byte_id_[static_cast<uint8_t>(info.token_piece[0])] = info.id;
    } else if (info.flag & kUnknown) {
      if (unk_id_ != kInvalidToken) {
        RETURN_ABORTED() << "bad format (too many unknown tokens)";
      }
      unk_id_ = info.id;
    }
  }

  // find id for space character
  auto it_space = token_dict_.find(" ");
  if (it_space == token_dict_.end()) {
    RETURN_ABORTED() << "bad format (no symbol for space)";
  }
  space_id_ = it_space->second->id;

  return OkStatus();
}

Status BpeModel::ReadRecord(ReadableFile *fp, TokenInfo *info) {
  info->flag = fp->readValue<int8_t>();

  // raw piece.
  int n_bytes = fp->readValue<uint8_t>();
  std::string piece;
  if (n_bytes) {
    piece = fp->readString(n_bytes);
  }
  info->token_piece = std::move(piece);
  if ((info->flag & kByte) && info->token_piece.size() != 1) {
    RETURN_ABORTED() << "bad format (byte)";
  }

  // piece display.
  n_bytes = fp->readValue<uint8_t>();
  std::string piece_display;
  if (n_bytes) {
    piece_display = fp->readString(n_bytes);
  }
  info->token_string = std::move(piece_display);

  // weight.
  info->weight = fp->readValue<float>();

  return OkStatus();
}

Status BpeModel::CheckModel() {
  for (int ch = 0; ch < 256 && byte_token_available_; ++ch) {
    if (byte_id_[ch] == kInvalidToken)
      RETURN_ABORTED() << "bad format, byte " << ch << " not exist in model";
  }

  if (unk_id_ == kInvalidToken) {
    RETURN_ABORTED() << "bad model (no unknown token)";
  }

  return OkStatus();
}

const std::string &BpeModel::token_string(int token_id) const {
  CHECK(token_id >= 0 && token_id < vocab_size());

  return tokens_[token_id].token_string;
}

const std::string &BpeModel::token_piece(int token_id) const {
  CHECK(token_id >= 0 && token_id < vocab_size());

  return tokens_[token_id].token_piece;
}

int BpeModel::FindToken(const std::string &token) const {
  auto it = token_dict_.find(token);
  if (it == token_dict_.end()) {
    return unk_id();
  }

  return it->second->id;
}

int BpeModel::byte_id(int ord) const {
  CHECK(ord >= 0 && ord < 256);
  return byte_id_[ord];
}

int BpeModel::vocab_size() const {
  return tokens_.size();
}

int BpeModel::unk_id() const {
  return unk_id_;
}

int BpeModel::space_id() const {
  return space_id_;
}

bool BpeModel::is_special(int token_id) const {
  return tokens_[token_id].is_special_token();
}

int32_t BpeModel::flag(int token_id) const {
  return tokens_[token_id].flag;
}

std::pair<int, float> BpeModel::FindMerge(int left, int right) const {
  std::string left_tok = tokens_[left].token_piece;
  std::string right_tok = tokens_[right].token_piece;
  std::string tok = left_tok + right_tok;

  auto it = token_dict_.find(tok);
  if (it == token_dict_.end()) {
    return std::make_pair(Vocab::kInvalidToken,
                          std::numeric_limits<float>::infinity());
  } else {
    return std::make_pair(it->second->id, -it->second->weight);
  }
}


// -- BpeEncoder ---------------------------------------------------------------

// String to BPE token-ids encoder.
class BpeEncoder : private util::NonCopyable {
 public:
  BpeEncoder(const BpeModel *model, const BpeConfig &config);

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
  const BpeConfig *config_;
  util::Pool<Symbol, kSymbolPoolBlockSize> symbol_pool_;
  Symbol *header_;
  std::priority_queue<Bigram, std::vector<Bigram>, std::greater<Bigram>> queue_;

  // initialize the symbol linked list from string `s` and store the pointer of
  // header node to `header_`.
  void InitSymbolList(const std::string &s);

  // initialize the queue by putting all possible two-bytes bigram to queue
  void InitQueue();

  // append a token to the tail of symbol linked-list. Returns the new tail
  // pointer.
  Symbol *AppendToken(Symbol *tail, int token_id);

  // split string `s` into a list of single-byte strings.
  std::vector<std::string> SplitBytes(const std::string &s);

  // add bigram (left, right) to queue if token left+right exists
  void AddBigramIfExist(Symbol *left, Symbol *right);

  // merge bigram (left, right) into one symbol, then clear original left and
  // right symbols and return pointer to the merged one.
  Symbol *Merge(const Bigram &bigram);

  // get the final symbol list from linked list pointered by header_
  std::vector<int> GetSymbolList();
};

BpeEncoder::BpeEncoder(const BpeModel *model, const BpeConfig &config)
    : model_(model),
      config_(&config),
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

std::vector<std::string> BpeEncoder::SplitBytes(const std::string &s) {
  std::vector<std::string> l;
  
  char buffer[2] = " ";
  for (char ch : s) {
    buffer[0] = ch;
    l.emplace_back(buffer);
  }

  return l;
}

BpeEncoder::Symbol *BpeEncoder::AppendToken(Symbol *tail, int token_id) {
  Symbol *symbol = symbol_pool_.Alloc();

  symbol->token_id = token_id;
  symbol->prev = tail;
  symbol->next = nullptr;

  tail->next = symbol;
  
  return symbol;
}

void BpeEncoder::InitSymbolList(const std::string &s) {
  // preprocess sentence
  Symbol *header = symbol_pool_.Alloc();
  header->prev = nullptr;
  header->token_id = Vocab::kInvalidToken;

  // prefix (_)
  Symbol *prefix = nullptr;
  if (config_->add_prefix_space) {
    prefix = AppendToken(header, model_->space_id());
  } else {
    prefix = header;
  }

  Symbol *prev = prefix;
  std::vector<std::string> initial_pieces;
  if (config_->split_by_unicode) {
    initial_pieces = strings::SplitToUtf8Chars(s);
  } else {
    initial_pieces = SplitBytes(s);
  }

  for (const std::string &utf8_ch : initial_pieces) {
    int token_id = utf8_ch == " " ? model_->space_id()
                                  : model_->FindToken(utf8_ch);
    if (token_id == model_->unk_id() && model_->byte_token_available()) {
      // symbol not found in the vocab, but byte token available.
      // Then, fallback to byte tokens.
      for (char ch : utf8_ch) {
        prev = AppendToken(prev, model_->byte_id(static_cast<uint8_t>(ch)));
      }
    } else {
      prev = AppendToken(prev, token_id);
    }
  }

  header_ = header;
}

// -- class BpeTokenizer -------------------------------------------------------

BpeTokenizer::BpeTokenizer() {}

expected_ptr<BpeTokenizer> BpeTokenizer::FromConfig(const BpeConfig &config) {
  auto model = BpeModel::FromModel(config.model_file);
  RETURN_IF_ERROR(model);

  std::unique_ptr<BpeTokenizer> tokenizer{new BpeTokenizer()};
  tokenizer->model_ = std::move(model).unique_ptr();
  tokenizer->config_ = config;

  return tokenizer;
}

std::vector<int> BpeTokenizer::Encode(const std::string &s) const {
  BpeEncoder encoder(model_.get(), config_);
  return encoder.Encode(s);
}

const Vocab *BpeTokenizer::vocab() const {
  return model_.get();
}

// -- class Tokenizer ----------------------------------------------------------

expected_ptr<Tokenizer> Tokenizer::FromConfig(const IniSection &config) {
  std::string type = config.getString("type");
  if (type == "bpe") {
    auto bpe_config = BpeConfig::FromConfig(config);
    RETURN_IF_ERROR(bpe_config);

    return BpeTokenizer::FromConfig(*bpe_config);
  } else {
    RETURN_ABORTED() << "invalid tokenizer type: " << type;
  }
}

}  // namespace llama
