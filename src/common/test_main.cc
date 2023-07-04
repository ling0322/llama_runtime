#include "environment.h"
#include "test_helper.h"

using llama::Environment;

int main(int argc, char **argv) {
  Environment::init();

  int result = Catch::Session().run(argc, argv);
  Environment::destroy();

  return result;
}
