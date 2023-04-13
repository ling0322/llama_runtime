#include "env.h"
#include "test_helper.h"

using llama::Env;

int main(int argc, char **argv) {
  Env::Init();

  int result = Catch::Session().run(argc, argv);
  Env::Destroy();

  return result;
}
