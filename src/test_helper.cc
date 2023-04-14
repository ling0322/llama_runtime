#include "test_helper.h"
#include "tensor.h"

#include "readable_file.h"

namespace llama {
namespace test_helper {

bool IsNear(const TensorView2Df &a, const TensorView2Df &b, float eps) {
  if (a.shape0() != b.shape0() || a.shape1() != b.shape1()) {
    return false;
  }

  Tensor2Df x(a.shape0(), a.shape1());
  x.CopyFrom(a);
  x.Add(-1.0f, b);
  x.ApplyAbs();
  return x.Max() <= eps;
}

bool IsNear(const TensorView &a, const TensorView &b, float eps) {
  if (a.dtype() != b.dtype() || a.rank() != b.rank()) {
    return false;
  }

  if (a.dtype() == DType::kFloat && a.rank() == 2) {
    return IsNear(a.ToTensorView2Df(), b.ToTensorView2Df(), eps);
  } else {
    return false;
  }
}

template<typename T>
Tensor2D<T> ReadTensor2D(const std::string &filename) {
  Tensor tensor;
  auto fp = ReadableFile::Open(filename);
  REQUIRE(fp.ok());

  Status status = tensor.Read(fp.get());
  REQUIRE(status.ok());

  status = tensor.EnsureRankDType(2, TypeID<T>());
  REQUIRE(status.ok());

  return tensor;
}

template Tensor2Df ReadTensor2D(const std::string &filename);

}  // namespace
}  // namespace llama
