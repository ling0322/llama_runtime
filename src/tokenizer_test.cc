#include "test_helper.h"
#include "path.h"
#include "strings.h"
#include "status.h"
#include "tokenizer.h"

using namespace llama;

TEST_CASE("llama tokenizer works", "[core][text]") {
  auto tokenizer = LlamaTokenizer::FromModel("data/test/tokenizer.lrtok");
  REQUIRE(tokenizer.ok());

  std::vector<std::string> pieces = tokenizer->EncodeAsPieces("hello world");
}
