#include "test_helper.h"
#include "path.h"
#include "io.h"
#include "strings.h"
#include "status.h"
#include "tokenizer.h"

using namespace llama;

TEST_CASE("llama tokenizer works", "[core][text]") {
  auto tokenizer = LlamaTokenizer::FromModel("data/test/tokenizer.lrtok");
  REQUIRE(tokenizer.ok());

  auto fp = ReadableFile::Open("data/test/tokenizer.lrtok");
}
