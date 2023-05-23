#include "bpe_model.h"

#include <stdint.h>
#include <math.h>
#include <functional>

#include "strings.h"
#include "strings.h"

namespace llama {

// -- class SpmBpeModel --------------------------------------------------------

SpmBpeModel::SpmBpeModel(): unk_id_(kInvalidToken) {
  for (int i = 0; i < 256; ++i) {
    byte_id_[i] = kInvalidToken;
  }
}

StatusOr<SpmBpeModel> SpmBpeModel::FromFile(const std::string &filename) {
  std::unique_ptr<SpmBpeModel> model(new SpmBpeModel());
  auto fp = ReadableFile::Open(filename);
  RETURN_IF_ERROR(fp) << "create LlamaTokenizer failed";

  RETURN_IF_ERROR(model->ReadModel(fp.get()));
  RETURN_IF_ERROR(model->CheckModel());
  return model;
}

Status SpmBpeModel::ReadModel(ReadableFile *fp) {
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
    RETURN_IF_ERROR(ReadRecord(fp, &info));
    info.id = token_id;

    tokens_.emplace_back(std::move(info));
  }

  RETURN_IF_ERROR(ReadMagicNumber(fp));
  RETURN_IF_ERROR(InitModel());
  return OkStatus();
}

Status SpmBpeModel::ReadMagicNumber(ReadableFile *fp) {
  // ensure magic number
  int16_t magic_number = 0;
  RETURN_IF_ERROR(fp->ReadValue(&magic_number));
  if (magic_number != kMagicNumber) {
    RETURN_ABORTED() << "bad format (magic number)";
  }

  return OkStatus();
}

Status SpmBpeModel::InitModel() {
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

Status SpmBpeModel::ReadRecord(ReadableFile *fp, TokenInfo *info) {
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

Status SpmBpeModel::CheckModel() {
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

const std::string &SpmBpeModel::token_string(int token_id) const {
  CHECK(token_id >= 0 && token_id < vocab_size());

  return tokens_[token_id].token_string;
}

int SpmBpeModel::FindToken(const std::string &token) const {
  auto it = token_dict_.find(token);
  if (it == token_dict_.end()) {
    return unk_id();
  }

  return it->second->id;
}

int SpmBpeModel::byte_id(int ord) const {
  CHECK(ord >= 0 && ord < 256);
  return byte_id_[ord];
}

int SpmBpeModel::vocab_size() const {
  return tokens_.size();
}

int SpmBpeModel::unk_id() const {
  return unk_id_;
}

int SpmBpeModel::space_id() const {
  return underscore_id_;
}

bool SpmBpeModel::is_special(int token_id) const {
  return tokens_[token_id].is_special_token();
}

int32_t SpmBpeModel::flag(int token_id) const {
  return tokens_[token_id].flag;
}

std::pair<int, float> SpmBpeModel::FindMerge(int left, int right) const {
  std::string left_tok = tokens_[left].token_string;
  std::string right_tok = tokens_[right].token_string;
  std::string tok = left_tok + right_tok;

  auto it = token_dict_.find(tok);
  if (it == token_dict_.end()) {
    return std::make_pair(Vocab::kInvalidToken,
                          std::numeric_limits<float>::infinity());
  } else {
    return std::make_pair(it->second->id, -it->second->weight);
  }
}

}  // namespace llama
