#include "test_helper.h"
#include "reader.h"
#include "strings.h"
#include "status.h"
#include "tokenizer.h"
#include "util.h"

using namespace llama;

TEST_CASE("llama tokenizer works", "[core][text]") {
  auto tokenizer = LlamaTokenizer::FromModel("data/test/tokenizer.lrtok");
  LOG(INFO) << tokenizer.what();
  REQUIRE(tokenizer.ok());

  auto fp = ReadableFile::Open("data/test/tokenizer.test_case.txt");
  std::string line;
  while (IsOK(fp->ReadLine(&line))) {
    line = strings::TrimRight(line, "\r\n");
    auto row = strings::Split(line, "\t");
    REQUIRE(row.size() == 2);

    auto sentence = std::move(row[0]);
    auto ref_pieces = strings::Split(strings::Trim(row[1]), " ");
    auto pieces = tokenizer->EncodeAsPieces(sentence);

    REQUIRE(pieces == ref_pieces);
  }
}
