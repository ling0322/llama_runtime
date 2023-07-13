#include "catch2/catch_amalgamated.hpp"
#include "flint/operators.h"
#include "flint/util.h"
#include "llmpp/chatglm2_model.h"
#include "llyn/ini_config.h"

using namespace llmpp;
using namespace ly;
using namespace flint;

TEST_CASE("test ChatGLM2 module", "[core][nn][chatglm2]") {
  ly::Path model_dir = ly::Path("data") / "test";
  Context ctx = getCtxForCPU();

  ly::Path config_file = model_dir / "chatglm2.config.ini";
  auto ini = IniConfig::read(config_file.string());
  auto config = ChatGLM2Config::fromIni(*ini);
  auto model = ChatGLM2Model::create(ctx, *config);

  ly::Path model_path = ini->getSection(kModelSection).getPath("params_file");
  readParameters(model_path.string(), model.get());

  Tensor in = Tensor::create<LongType>({1, 5}, {
      64790, 64792, 790, 30951, 517
  });
  Tensor x = model->forward(nullptr, in);
  ctx.F()->print(x);
}
