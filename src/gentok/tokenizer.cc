#include "gentok/tokenizer.h"

#include <stdint.h>
#include <math.h>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "llyn/error.h"
#include "llyn/ini_config.h"
#include "llyn/pool.h"
#include "llyn/reader.h"
#include "llyn/strings.h"


using ly::AbortedError;
using ly::IniConfig;
using ly::IniSection;
using ly::ReadableFile;

namespace gentok {

// config for BPE tokenizer.
struct BPEConfig {
  // path of the BPE model.
  std::string modelFile;

  // true if add a space into the begining of text.
  bool addPrefixSpace;

  // true if split by unicode characters before merging. false if split by byte.
  bool splitByUnicode;

  // create the BPE config from ini config.
  static std::unique_ptr<BPEConfig> fromIni(const IniSection &config);

  // contructor for the default config.
  BPEConfig();
};

// model for BPE tokenizer.
class BPEModel;

// BPE tokenizer.
class BPETokenizer : public Tokenizer {
 public:
  static std::unique_ptr<BPETokenizer> create(const BPEConfig &config);

  // implement interface Tokenizer
  std::vector<int> encode(const std::string &s) const override;
  const Vocab *getVocab() const override;

 private:
  std::unique_ptr<BPEModel> _model;
  BPEConfig _config;

  BPETokenizer();
};

// -- BPEConfig ----------

BPEConfig::BPEConfig()
    : addPrefixSpace(true),
      splitByUnicode(true) {}

std::unique_ptr<BPEConfig> BPEConfig::fromIni(const IniSection &iniSection) {
  std::unique_ptr<BPEConfig> config = std::make_unique<BPEConfig>();

  config->addPrefixSpace = iniSection.getBool("add_prefix_space");
  config->splitByUnicode = iniSection.getBool("split_by_unicode");
  config->modelFile = iniSection.getPath("model_file").string();

  return config;
}

// -- BpeModel ----------

// Store tne data from sentence-piece BPE model.
class BPEModel : public Vocab, private ly::NonCopyable {
 public:
  // token flags.
  static constexpr int kUnknown = 1;
  static constexpr int kControl = 2;
  static constexpr int kByte = 4;
  static constexpr int kUnused = 8;

  // Read the BpeModel from file. It could not read a SPM model directly, instead, we need to
  // convert the SPM model using tokenizer_exporter.py.
  static std::unique_ptr<BPEModel> create(const std::string &filename);

  // implement interface Vocab
  int findToken(const std::string &token) const override;
  const std::string &getTokenPiece(int token_id) const override;
  const std::string &getTokenString(int token_id) const override;
  int getVocabSize() const override;
  int getUnkId() const override;

  // given the left and right token-ids, find the merged token-id and cost, return as (id, cost).
  // If no merge record found, return (Vocab::kInvalidToken, +inf)
  std::pair<int, float> findMerge(int left, int right) const;

  // token-id for space token.
  int getSpaceId() const;

  // return true if the token is either Unknown, Control, SingleByte or Unsed.
  bool isSpecialToken(int token_id) const;

  // returns flag of a token.
  int32_t getTokenFlag(int token_id) const;

  // get token-id for a single byte.
  int getByteId(int byte) const;
  bool isByteTokenAvailable() const { return _isByteTokenAvailable; }

 private:
  struct TokenInfo;

  static constexpr int16_t kMagicNumber = 0x55aa;

  std::unordered_map<std::string, const TokenInfo *> _tokenDict;
  std::vector<TokenInfo> _tokens;

  // standalone tokens for byte.
  std::array<int, 256> _byteId;
  bool _isByteTokenAvailable;

  int _unkId;
  int _spaceId;
  
  BPEModel();

  // read model from fp
  void readModel(ReadableFile *fp);
  void readMagicNumber(ReadableFile *fp);
  TokenInfo readRecord(ReadableFile *fp);
  void initModel();
  void checkModel();
};

struct BPEModel::TokenInfo {
  int id;
  float weight;
  std::string tokenPiece;
  std::string tokenString;
  int8_t flag;

  constexpr bool isSpecialToken() const { return flag != 0; }
};


BPEModel::BPEModel() : _unkId(kInvalidToken), _isByteTokenAvailable(false) {
  std::fill(_byteId.begin(), _byteId.end(), kInvalidToken);
}

std::unique_ptr<BPEModel> BPEModel::create(const std::string &filename) {
  std::unique_ptr<BPEModel> model(new BPEModel());
  auto fp = ReadableFile::open(filename);

  model->readModel(fp.get());
  model->checkModel();
  return model;
}

void BPEModel::readModel(ReadableFile *fp) {
  std::string s = fp->readString(4);
  if (s != "LLsp") {
    throw AbortedError("bad format (header)");
  }

  int32_t numTokens = fp->readValue<int32_t>();
  readMagicNumber(fp);

  // read the list of token info
  _tokens.clear();
  for (int tokenId = 0; tokenId < numTokens; ++tokenId) {
    TokenInfo info = readRecord(fp);
    info.id = tokenId;

    _tokens.emplace_back(std::move(info));
  }

  readMagicNumber(fp);
  initModel();
}

void BPEModel::readMagicNumber(ReadableFile *fp) {
  // ensure magic number
  int16_t magic_number = fp->readValue<int16_t>();
  if (magic_number != kMagicNumber) {
    throw AbortedError("bad format (magic number)");
  }
}

void BPEModel::initModel() {
  // build token_dict_ and byte_id_
  _tokenDict.clear();
  

  for (const TokenInfo &info : _tokens) {
    if (!info.flag) {
      // flag == 0 means it's a normal token (not control token, unknown token
      // or unused token)
      _tokenDict[info.tokenPiece] = &info;
    } else if (info.flag & kByte) {
      // single byte token
      CHECK(info.tokenPiece.size() == 1);
      _isByteTokenAvailable = true;
      _byteId[static_cast<uint8_t>(info.tokenPiece[0])] = info.id;
    } else if (info.flag & kUnknown) {
      if (_unkId != kInvalidToken) {
        throw AbortedError("bad format (too many unknown tokens)");
      }
      _unkId = info.id;
    }
  }

  // find id for space character
  auto itSpace = _tokenDict.find(" ");
  if (itSpace == _tokenDict.end()) {
    throw AbortedError("bad format (no symbol for space)");
  }
  _spaceId = itSpace->second->id;
}

BPEModel::TokenInfo BPEModel::readRecord(ReadableFile *fp) {
  TokenInfo info;
  info.flag = fp->readValue<int8_t>();

  // raw piece.
  int nBytes = fp->readValue<uint8_t>();
  std::string piece;
  if (nBytes) {
    piece = fp->readString(nBytes);
  }
  info.tokenPiece = std::move(piece);
  if ((info.flag & kByte) && info.tokenPiece.size() != 1) {
    throw AbortedError("bad format (byte)");
  }

  // piece display.
  nBytes = fp->readValue<uint8_t>();
  std::string pieceDisplay;
  if (nBytes) {
    pieceDisplay = fp->readString(nBytes);
  }
  info.tokenString = std::move(pieceDisplay);

  // weight.
  info.weight = fp->readValue<float>();

  return info;
}

void BPEModel::checkModel() {
  for (int ch = 0; ch < 256 && _isByteTokenAvailable; ++ch) {
    if (_byteId[ch] == kInvalidToken)
      throw AbortedError(ly::sprintf("bad format, byte %d not exist in model", ch));
  }

  if (_unkId == kInvalidToken) {
    throw AbortedError("bad model (no unknown token)");
  }
}

const std::string &BPEModel::getTokenString(int tokenId) const {
  CHECK(tokenId >= 0 && tokenId < getVocabSize());

  return _tokens[tokenId].tokenString;
}

const std::string &BPEModel::getTokenPiece(int tokenId) const {
  CHECK(tokenId >= 0 && tokenId < getVocabSize());

  return _tokens[tokenId].tokenPiece;
}

int BPEModel::findToken(const std::string &token) const {
  auto it = _tokenDict.find(token);
  if (it == _tokenDict.end()) {
    return getUnkId();
  }

  return it->second->id;
}

int BPEModel::getByteId(int ord) const {
  CHECK(ord >= 0 && ord < 256);
  return _byteId[ord];
}

int BPEModel::getVocabSize() const {
  return _tokens.size();
}

int BPEModel::getUnkId() const {
  return _unkId;
}

int BPEModel::getSpaceId() const {
  return _spaceId;
}

bool BPEModel::isSpecialToken(int token_id) const {
  return _tokens[token_id].isSpecialToken();
}

int32_t BPEModel::getTokenFlag(int token_id) const {
  return _tokens[token_id].flag;
}

std::pair<int, float> BPEModel::findMerge(int left, int right) const {
  std::string left_tok = _tokens[left].tokenPiece;
  std::string right_tok = _tokens[right].tokenPiece;
  std::string tok = left_tok + right_tok;

  auto it = _tokenDict.find(tok);
  if (it == _tokenDict.end()) {
    return std::make_pair(Vocab::kInvalidToken, std::numeric_limits<float>::infinity());
  } else {
    return std::make_pair(it->second->id, -it->second->weight);
  }
}


// -- BpeEncoder ----------

// String to BPE token-ids encoder.
class BPEEncoder : private ly::NonCopyable {
 public:
  BPEEncoder(const BPEModel *model, const BPEConfig &config);

  // encode string to token ids. 
  std::vector<int> encode(const std::string &s);

 private:
  static constexpr int kSymbolPoolBlockSize = 256;

  // symbol linked list
  struct Symbol {
    Symbol *prev;
    Symbol *next;
    int tokenId;

    bool valid() const { return tokenId != Vocab::kInvalidToken; }
  };

  struct Bigram {
    Symbol *left;
    Symbol *right;
    float cost;
    int mergedTokenId;

    bool operator>(const Bigram &rhs) const {
      return cost > rhs.cost;
    }
  };

  const BPEModel *_model;
  const BPEConfig *_config;
  ly::Pool<Symbol, kSymbolPoolBlockSize> _symbolPool;
  Symbol *_header;
  std::priority_queue<Bigram, std::vector<Bigram>, std::greater<Bigram>> _queue;

  // initialize the symbol linked list from string `s` and store the pointer of header node to
  // `header_`.
  void initSymbolList(const std::string &s);

  // initialize the queue by putting all possible two-bytes bigram to queue
  void initQueue();

  // append a token to the tail of symbol linked-list. Returns the new tail pointer.
  Symbol *appendToken(Symbol *tail, int token_id);

  // split string `s` into a list of single-byte strings.
  std::vector<std::string> splitBytes(const std::string &s);

  // add bigram (left, right) to queue if token left+right exists
  void addBigramIfExist(Symbol *left, Symbol *right);

  // merge bigram (left, right) into one symbol, then clear original left and right symbols and
  // return pointer to the merged one.
  Symbol *mergeBigram(const Bigram &bigram);

  // get the final symbol list from linked list pointered by header_
  std::vector<int> getSymbolList();
};

BPEEncoder::BPEEncoder(const BPEModel *model, const BPEConfig &config)
    : _model(model),
      _config(&config),
      _header(nullptr) {}

void BPEEncoder::initQueue() {
  Symbol *p = _header->next,
         *q = p->next;
  while (q) {
    addBigramIfExist(p, q);
    p = q;
    q = q->next;
  }
}

std::vector<int> BPEEncoder::getSymbolList() {
  std::vector<int> tokenIds;
  Symbol *p = _header->next;
  while (p) {
    tokenIds.push_back(p->tokenId);
    p = p->next;
  }

  return tokenIds;
}

std::vector<int> BPEEncoder::encode(const std::string &s) {
  initSymbolList(s);
  initQueue();

  // loop until there is no bigram candidates
  while (!_queue.empty()) {
    Bigram bigram = _queue.top();
    _queue.pop();

    if (bigram.left->valid() && bigram.right->valid()) {
      Symbol *symbol = mergeBigram(bigram);
      addBigramIfExist(symbol->prev, symbol);
      addBigramIfExist(symbol, symbol->next);
    }
  }

  return getSymbolList();
}

void BPEEncoder::addBigramIfExist(Symbol *left, Symbol *right) {
  if (left == _header || right == nullptr ||
      _model->isSpecialToken(right->tokenId) ||
      _model->isSpecialToken(left->tokenId)) {
    return;
  }

  int mergedTokenId;
  float cost;
  std::tie(mergedTokenId, cost) = _model->findMerge(left->tokenId, right->tokenId);
  if (mergedTokenId == Vocab::kInvalidToken) {
    return;
  }

  Bigram bigram;
  bigram.left = left;
  bigram.right = right;
  bigram.cost = cost;
  bigram.mergedTokenId = mergedTokenId;
  _queue.push(bigram);
}

BPEEncoder::Symbol *BPEEncoder::mergeBigram(const Bigram &bigram) {
  Symbol *left = bigram.left;
  Symbol *right = bigram.right;
  Symbol *next = right->next;
  Symbol *prev = left->prev;

  Symbol *merged = _symbolPool.alloc();
  merged->tokenId = bigram.mergedTokenId;
  merged->next = next;
  merged->prev = prev;
  if (next) {
    next->prev = merged;
  }
  // prev do not need to check since there is a header node
  prev->next = merged;

  right->tokenId = Vocab::kInvalidToken;
  right->next = nullptr;
  right->prev = nullptr;

  left->tokenId = Vocab::kInvalidToken;
  left->next = nullptr;
  left->prev = nullptr;

  return merged;
}

std::vector<std::string> BPEEncoder::splitBytes(const std::string &s) {
  std::vector<std::string> l;
  
  char buffer[2] = " ";
  for (char ch : s) {
    buffer[0] = ch;
    l.emplace_back(buffer);
  }

  return l;
}

BPEEncoder::Symbol *BPEEncoder::appendToken(Symbol *tail, int tokenId) {
  Symbol *symbol = _symbolPool.alloc();

  symbol->tokenId = tokenId;
  symbol->prev = tail;
  symbol->next = nullptr;

  tail->next = symbol;
  
  return symbol;
}

void BPEEncoder::initSymbolList(const std::string &s) {
  // preprocess sentence
  Symbol *header = _symbolPool.alloc();
  header->prev = nullptr;
  header->tokenId = Vocab::kInvalidToken;

  // prefix (_)
  Symbol *prefix = nullptr;
  if (_config->addPrefixSpace) {
    prefix = appendToken(header, _model->getSpaceId());
  } else {
    prefix = header;
  }

  Symbol *prev = prefix;
  std::vector<std::string> initialPieces;
  if (_config->splitByUnicode) {
    initialPieces = ly::splitUtf8(s);
  } else {
    initialPieces = splitBytes(s);
  }

  for (const std::string &piece : initialPieces) {
    int tokenId = piece == " " ? _model->getSpaceId() : _model->findToken(piece);
    if (tokenId == _model->getUnkId() && _model->isByteTokenAvailable()) {
      // symbol not found in the vocab, but byte token available.
      // Then, fallback to byte tokens.
      for (char ch : piece) {
        prev = appendToken(prev, _model->getByteId(static_cast<uint8_t>(ch)));
      }
    } else {
      prev = appendToken(prev, tokenId);
    }
  }

  _header = header;
}

// -- class BPETokenizer ----------

BPETokenizer::BPETokenizer() {}

std::unique_ptr<BPETokenizer> BPETokenizer::create(const BPEConfig &config) {
  auto model = BPEModel::create(config.modelFile);

  std::unique_ptr<BPETokenizer> tokenizer{new BPETokenizer()};
  tokenizer->_model = std::move(model);
  tokenizer->_config = config;

  return tokenizer;
}

std::vector<int> BPETokenizer::encode(const std::string &s) const {
  BPEEncoder encoder(_model.get(), _config);
  return encoder.encode(s);
}

const Vocab *BPETokenizer::getVocab() const {
  return _model.get();
}

// -- class Tokenizer ----------

std::unique_ptr<Tokenizer> Tokenizer::create(const IniSection &config) {
  std::string type = config.getString("type");
  if (type == "bpe") {
    auto bpe_config = BPEConfig::fromIni(config);

    return BPETokenizer::create(*bpe_config);
  } else {
    throw AbortedError(ly::sprintf("invalid tokenizer type: %s", type));
  }
}

}  // namespace gentok
