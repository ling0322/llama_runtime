#include "test_helper.h"
#include "reader.h"
#include "strings.h"
#include "status.h"
#include "tokenizer.h"
#include "util.h"

using namespace llama;

std::vector<std::string> EncodeAsPieces(
    const Tokenizer *tokenizer,
    const std::string &s) {
  std::vector<std::string> tokens;
  std::vector<int> token_ids = tokenizer->Encode(s);

  const Vocab *vocab = tokenizer->vocab();
  for (int token_id : token_ids) {
    tokens.emplace_back(vocab->token_string(token_id));
  }
  return tokens;
}

TEST_CASE("llama tokenizer works", "[core][tokenizer]") {
  BpeConfig bpe_config;
  bpe_config.add_prefix_space = false;
  bpe_config.split_by_unicode = false;
  bpe_config.model_file = "data/test/gpt2_bpe.tokenizer.bin";

  auto tokenizer = BpeTokenizer::FromConfig(bpe_config);
  REQUIRE(tokenizer.ok());

  auto fp = ReadableFile::Open("data/test/gpt2_bpe.tokenizer.test_cases.txt");
  REQUIRE(fp.ok());

  std::string line;
  while (IsOK(fp->ReadLine(&line))) {
    line = strings::TrimRight(line, "\r\n");
    auto row = strings::Split(line, "\t");
    REQUIRE(row.size() == 2);

    auto sentence = std::move(row[0]);
    auto ref_pieces = strings::Split(strings::Trim(row[1]), " ");
    auto pieces = EncodeAsPieces(tokenizer.get(), sentence);

    REQUIRE(pieces == ref_pieces);
  }
}
