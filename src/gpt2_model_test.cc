#include "test_helper.h"
#include "nn_test_helper.h"
#include "gpt2_model.h"

#include "ini_config.h"
#include "operators.h"
#include "util.h"

using namespace llama;
using namespace nn;

TEST_CASE("test GPT2 module", "[core][nn][gpt2]") {
  util::Path model_dir = util::Path("data") / "test";
  Context ctx = MustGetCtxForCPU();

  util::Path config_file = model_dir / "gpt2.config.ini";
  auto ini = IniConfig::Read(config_file.string());
  REQUIRE_OK(ini);

  auto config = GPT2Config::FromIni(*ini);
  REQUIRE_OK(config);

  auto model = GPT2Model::Create(ctx, *config);
  REQUIRE_OK(model);

  util::Path model_path = "";
  Status status = ini->section(kModelSection).Get("params_file", &model_path);
  MustReadParameters(model_path.string(), model.get());

  util::Path tensor_file = model_dir / "gpt2.test_tensors.bin";
  std::vector<Tensor> tensors = MustReadAllTensors(tensor_file.string());
  REQUIRE(tensors.size() == 2);

  Tensor in = tensors[0];
  Tensor out = tensors[1];
  Tensor x = model->Forward(nullptr, in);
  x = model->Logits(x);

  REQUIRE(ctx.F()->AllClose(out, x));

  // autoregressive mode
  TensorMap kv_cache;
  x = in.Slice(1, 0, 5);  // tensor x is in NWC format
  x = model->Forward(&kv_cache, x);
  Tensor o = model->Logits(x);
  for (int i = 5; i < in.shape(1); ++i) {
    x = in.Slice(1, i, i + 1);
    x = model->Forward(&kv_cache, x);
    x = model->Logits(x);
    o = ctx.F()->Cat(o, x, 1);
  }

  REQUIRE(ctx.F()->AllClose(o, out));
}
