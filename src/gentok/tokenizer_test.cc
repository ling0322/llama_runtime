#include "catch2/catch_amalgamated.hpp"
#include "llyn/ini_config.h"
#include "llyn/reader.h"
#include "llyn/strings.h"
#include "gentok/tokenizer.h"

using namespace gentok;
using namespace ly;


std::vector<std::string> encodeAsPieces(const Tokenizer *tokenizer, const std::string &s) {
  std::vector<std::string> tokens;
  std::vector<int> tokenIds = tokenizer->encode(s);

  const Vocab *vocab = tokenizer->getVocab();
  for (int tokenId : tokenIds) {
    tokens.emplace_back(vocab->getTokenString(tokenId));
  }
  return tokens;
}

void testTokenizer(const std::string &ini_file, const std::string &test_case) {
  auto config = IniConfig::read(ini_file);
  const IniSection &section = config->getSection("tokenizer");

  auto tokenizer = Tokenizer::create(section);
  auto fp = ReadableFile::open(test_case);
  Scanner scanner(fp.get());
  while (scanner.scan()) {
    std::string line = ly::trimRight(scanner.getText(), "\r\n");
    auto row = ly::split(line, "\t");
    REQUIRE(row.size() == 2);

    auto sentence = std::move(row[0]);
    auto ref_pieces = ly::split(ly::trim(row[1]), " ");
    auto pieces = encodeAsPieces(tokenizer.get(), sentence);

    REQUIRE(pieces == ref_pieces);
  }
}

TEST_CASE("llama tokenizer works", "[core][tokenizer]") {
  testTokenizer("data/test/gpt2_bpe.tokenizer.ini",
                "data/test/gpt2_bpe.tokenizer.test_cases.txt");

  testTokenizer("data/test/llama_spm.tokenizer.ini",
                "data/test/llama_spm.tokenizer.test_cases.txt");
}
