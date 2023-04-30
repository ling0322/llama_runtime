#include "environment.h"
#include "test_helper.h"

using llama::Environment;

int main(int argc, char **argv) {
  Environment::Init();

  int result = Catch::Session().run(argc, argv);
  Environment::Destroy();

  return result;
}
