#pragma once

#include <vector>
#include "flint/nn.h"
#include "llyn/span.h"

namespace flint {

template<typename T>
std::vector<T> repeat(ly::Span<const T> v, int n) {
  std::vector<T> rep;
  for (int i = 0; i < n; ++i) {
    for (const T &elem : v) {
      rep.emplace_back(elem);
    }
  }

  return rep;
}

void readParameters(const std::string &model_path, Module *module);
std::vector<Tensor> readAllTensors(const std::string &filename);
Context getCtxForCPU();

}
