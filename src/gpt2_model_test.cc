#include "test_helper.h"
#include "nn_test_helper.h"
#include "gpt2_model.h"

#include "ini_parser.h"
#include "operators.h"
#include "util.h"

using namespace llama;
using namespace nn;

TEST_CASE("test GPT2 module", "[core][nn][gpt2]") {
  util::Path model_dir = util::Path("data") / "test";
  Context ctx = MustGetCtxForCPU();

  util::Path config_file = model_dir / "gpt2.config.ini";
  auto ini = IniParser::Read(config_file.string());
  REQUIRE(ini.ok());

  auto config = GPT2Config::FromIni(*ini);
  REQUIRE(config.ok());

  auto model = GPT2Model::Create(ctx, *config);
  REQUIRE(model.ok());

  util::Path model_path = "";
  Status status = ini->Get(kModelSection, "params_file", &model_path);
  MustReadParameters(model_path.string(), model.get());

  util::Path tensor_file = model_dir / "gpt2.test_tensors.bin";
  std::vector<Tensor> tensors = MustReadAllTensors(tensor_file.string());
  REQUIRE(tensors.size() == 2);

  Tensor in = tensors[0];
  Tensor out = tensors[1];
  Tensor x = model->Forward(nullptr, in);
  x = model->Logits(x);

  REQUIRE(ctx.F()->AllClose(out, x));
}
