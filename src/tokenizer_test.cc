#include "test_helper.h"
#include "ini_config.h"
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

void TestTokenizer(const std::string &ini_file, const std::string &test_case) {
  auto config = IniConfig::read(ini_file);
  const IniSection &section = config->getSection("tokenizer");

  auto tokenizer = Tokenizer::FromConfig(section);
  REQUIRE_OK(tokenizer);

  auto fp = ReadableFile::open(test_case);
  Scanner scanner(fp.get());
  while (scanner.scan()) {
    std::string line = strings::TrimRight(scanner.getText(), "\r\n");
    auto row = strings::Split(line, "\t");
    REQUIRE(row.size() == 2);

    auto sentence = std::move(row[0]);
    auto ref_pieces = strings::Split(strings::Trim(row[1]), " ");
    auto pieces = EncodeAsPieces(tokenizer.get(), sentence);

    REQUIRE(pieces == ref_pieces);
  }
}

TEST_CASE("llama tokenizer works", "[core][tokenizer]") {
  TestTokenizer("data/test/gpt2_bpe.tokenizer.ini",
                "data/test/gpt2_bpe.tokenizer.test_cases.txt");

  TestTokenizer("data/test/llama_spm.tokenizer.ini",
                "data/test/llama_spm.tokenizer.test_cases.txt");
}
