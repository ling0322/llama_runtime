#include "catch2/catch_amalgamated.hpp"
#include "llmrt/environment.h"

using llama::Environment;

int main(int argc, char **argv) {
  Environment::init();

  int result = Catch::Session().run(argc, argv);
  Environment::destroy();

  return result;
}
