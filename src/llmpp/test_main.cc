#include "catch2/catch_amalgamated.hpp"
#include "llmpp/environment.h"

using llmpp::Environment;

int main(int argc, char **argv) {
  Environment::init();

  int result = Catch::Session().run(argc, argv);
  Environment::destroy();

  return result;
}
