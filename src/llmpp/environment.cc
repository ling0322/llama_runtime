#include "llmpp/environment.h"

#include <mutex>
#include <thread>
#include "pmpack/pmpack.h"

namespace llmpp {

// -- class Environment::Impl ----------

class Environment::Impl {
 public:
  Impl();
  ~Impl();

 private:
  void init();
};

Environment::Impl::Impl() {
  init();
}
Environment::Impl::~Impl() {
  pmpack_destroy();
}

void Environment::Impl::init() {
  pmpack_init();
}

// -- class Environment::Impl ----------

Environment::Impl *Environment::_instance = nullptr;

void Environment::init() {
  _instance = new Impl();
}

void Environment::destroy() {
  delete _instance;
}

}  // namespace llama
