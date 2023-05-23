#ifndef LLMRUNTIME_BPE_MODEL_H_
#define LLMRUNTIME_BPE_MODEL_H_

#include <queue>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "reader.h"
#include "status.h"
#include "vocab.h"
#include "util.h"

namespace llama {

// The BPE tokenizer merge model. Given the left and right token-ids, return
// the cost or merged-token-id. 
class BpeModel : public Vocab {
 public:
  // token flags.
  static constexpr int kUnknown = 1;
  static constexpr int kControl = 2;
  static constexpr int kByte = 4;
  static constexpr int kUnused = 8;
  
  // given the left and right token-ids, find the merged token-id and cost,
  // return as (id, cost). If no merge record found, return
  // (Vocab::kInvalidToken, +inf)
  virtual std::pair<int, float> FindMerge(int left, int right) const = 0;

  // token-id for space token.
  virtual int space_id() const = 0;

  // return true if the token is either Unknown, Control, SingleByte or Unsed.
  virtual bool is_special(int token_id) const = 0;

  // returns flag of a token.
  virtual int32_t flag(int token_id) const = 0;

  // get token-id for a single byte.
  virtual int byte_id(int byte) const = 0;
};

// Store tne data from sentence-piece BPE model.
class SpmBpeModel : public BpeModel,
                    private util::NonCopyable {
 public:
  // Read the SpmBpeModel from file. It could not read a SPM model directly,
  // instead, we need to convert the SPM model using tokenizer_exporter.py.
  static StatusOr<SpmBpeModel> FromFile(const std::string &filename);

  // implement interface Vocab
  int FindToken(const std::string &token) const override;
  const std::string &token_string(int token_id) const override;
  int vocab_size() const override;
  int unk_id() const override;

  // implement interface BpeModel
  virtual std::pair<int, float> FindMerge(int left, int right) const override;
  virtual int space_id() const override;
  virtual bool is_special(int token_id) const override;
  virtual int32_t flag(int token_id) const override;
  virtual int byte_id(int byte) const override;

 private:
  struct TokenInfo;

  static constexpr int16_t kMagicNumber = 0x55aa;

  static constexpr char kUnderscore[] = "\xe2\x96\x81";

  std::unordered_map<std::string, const TokenInfo *> token_dict_;
  std::vector<TokenInfo> tokens_;
  int byte_id_[256];

  int unk_id_;
  int underscore_id_;
  
  SpmBpeModel();

  // read model from fp
  Status ReadModel(ReadableFile *fp);
  Status ReadMagicNumber(ReadableFile *fp);
  Status ReadRecord(ReadableFile *fp, TokenInfo *info);
  Status InitModel();
  Status CheckModel();
};

struct SpmBpeModel::TokenInfo {
  int id;
  float weight;
  std::string token_string;
  int token_byte;
  int8_t flag;

  constexpr bool is_special_token() const { return flag != 0; }
};

}  // namespace llama

#endif  // LLMRUNTIME_BPE_MODEL_H_
