#include "common/environment.h"

#include <mutex>
#include <thread>
#include "common/common.h"
#include "util/shared_library.h"
#include "pmpack/pmpack.h"
#include "util/log.h"
#include "util/util.h"

namespace llama {

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
