#include <chrono>
#include "common/test_helper.h"
#include "nn/nn_test_helper.h"
#include "model/gpt2_model.h"
#include "util/ini_config.h"
#include "nn/operators.h"
#include "util/strings.h"
#include "util/util.h"

using namespace llama;
using namespace nn;
using namespace std::literals;

TEST_CASE("benchmark GPT2 module", "[core][nn][gpt2]") {
  util::Path model_dir = util::Path("data") / "test";
  Context ctx = MustGetCtxForCPU();

  util::Path config_file = model_dir / "gpt2.config.ini";
  auto ini = IniConfig::read(config_file.string());
  auto config = GPT2Config::fromIni(*ini);
  auto model = GPT2Model::create(ctx, *config);

  util::Path model_path = ini->getSection(kModelSection).getPath("params_file");
  MustReadParameters(model_path.string(), model.get());

  util::Path tensor_file = model_dir / "gpt2.test_tensors.bin";
  std::vector<Tensor> tensors = MustReadAllTensors(tensor_file.string());
  REQUIRE(tensors.size() == 2);

  Tensor in = tensors[0];
  Tensor out = tensors[1];
  Tensor x = model->forward(nullptr, in);
  x = model->logits(x);

  REQUIRE(ctx.F()->allClose(out, x));

  // autoregressive mode
  TensorMap kv_cache;
  x = in.slice(1, 0, 5);  // tensor x is in NWC format
  auto t0 = std::chrono::high_resolution_clock::now();
  x = model->forward(&kv_cache, x);
  auto t1 = std::chrono::high_resolution_clock::now();
  auto delta = t1 - t0;
  auto duration_ms = delta / 1ns / 1e6f / 5;

  LOG(INFO) << str::sprintf("batch mode: %.2fms per token.", duration_ms);

  Tensor o = model->logits(x);
  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 5; i < in.getShape(1); ++i) {
    x = in.slice(1, i, i + 1);
    x = model->forward(&kv_cache, x);
    x = model->logits(x);
    o = ctx.F()->cat(o, x, 1);
  }
  t1 = std::chrono::high_resolution_clock::now();
  delta = t1 - t0;
  duration_ms = delta / 1ns / 1e6f / 5;

  LOG(INFO) << str::sprintf("auto regressive mode: %.2fms per token.", duration_ms);

  REQUIRE(ctx.F()->allClose(o, out));
}
