#include "catch2/catch_amalgamated.hpp"
#include "flint/operators.h"
#include "flint/util.h"
#include "llmpp/gpt2_model.h"
#include "llyn/ini_config.h"

using namespace llmpp;
using namespace ly;
using namespace flint;

TEST_CASE("test GPT2 module", "[core][nn][gpt2]") {
  ly::Path model_dir = ly::Path("data") / "test";
  Context ctx = getCtxForCPU();

  ly::Path config_file = model_dir / "gpt2.config.ini";
  auto ini = IniConfig::read(config_file.string());
  auto config = GPT2Config::fromIni(*ini);
  auto model = GPT2Model::create(ctx, *config);

  ly::Path model_path = ini->getSection(kModelSection).getPath("params_file");
  readParameters(model_path.string(), model.get());

  ly::Path tensor_file = model_dir / "gpt2.test_tensors.bin";
  std::vector<Tensor> tensors = readAllTensors(tensor_file.string());
  REQUIRE(tensors.size() == 2);

  Tensor in = tensors[0];
  Tensor out = tensors[1];
  Tensor x = model->forward(nullptr, in);
  x = model->logits(x);

  REQUIRE(ctx.F()->allClose(out, x));

  // autoregressive mode
  TensorMap kv_cache;
  x = in.slice(1, 0, 5);  // tensor x is in NWC format
  x = model->forward(&kv_cache, x);
  Tensor o = model->logits(x);
  for (int i = 5; i < in.getShape(1); ++i) {
    x = in.slice(1, i, i + 1);
    x = model->forward(&kv_cache, x);
    x = model->logits(x);
    o = ctx.F()->cat(o, x, 1);
  }

  REQUIRE(ctx.F()->allClose(o, out));
}
