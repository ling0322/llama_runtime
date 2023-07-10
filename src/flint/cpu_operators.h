#pragma once

#include <stdint.h>
#include <memory>
#include "flint/nn.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace flint {

// the CPU implementation of Operators
class CPUOperators : public Operators {
 public:
  CPUOperators();

  // create a instance of CPUOperators
  static std::unique_ptr<Operators> create();

  // implement interface Operators
  Tensor lookup(TensorCRef table, TensorCRef indices) override;
  Tensor matmul(TensorCRef a, TensorCRef b) override;
  Tensor mul(TensorCRef input, float other) override;
  Tensor softmax(TensorCRef input) override;
  Tensor gelu(TensorCRef input) override;
  Tensor add(TensorCRef a, TensorCRef b) override;
  Tensor createTensor(std::initializer_list<int> shape, DType dtype) override;
  Tensor createTensorLike(TensorCRef input) override;
  Tensor rand(std::initializer_list<int> shape, DType dtype) override;
  Tensor zeros(std::initializer_list<int> shape, DType dtype) override;
  Tensor contiguous(TensorCRef input) override;
  bool allClose(TensorCRef A, TensorCRef B) override;
  void print(TensorCRef tensor) override;
  Tensor layerNorm(TensorCRef input, TensorCRef weight, TensorCRef bias, float eps) override;
  Tensor causalMask(int max_len) override;
  Tensor cat(TensorCRef A, TensorCRef B, int dim) override;

 private:
  typedef TensorShape::Elem Shape;

  // sub-tensor. Stores the shape and data pointer to a sub region of the 
  // original Tensor. It's faster than Tensor when passing as parameters.
  template<typename T>
  struct Subtensor;

  template<typename T>
  class SubtensorList;

  // Get sub-tensor from Tensor.
  template<typename T>
  Subtensor<T> makeSubtensor(Tensor &tensor); 
  template<typename T>
  Subtensor<const T> makeConstSubtensor(const Tensor &tensor); 

  Tensor createTensor(ly::Span<const int> shape, DType dtype);
  Tensor createTensorLike(Subtensor<const float> A);

  Tensor matmulFp32(const Tensor &A, const Tensor &B);
  Tensor bmmFp32(const Tensor &A, const Tensor &B);
  Tensor bmmFp32QInt4Fp32(const Tensor &A, const Tensor &B);
  Tensor gemmFp32(const Tensor &A, const Tensor &B);
  Tensor gemmFp32QInt4Fp32(const Tensor &A, const Tensor &B);

  Tensor mulFp32(Subtensor<const float> A, float k);
  Tensor addFp32(Subtensor<const float> A, Subtensor<const float> B);
  Tensor softmaxFp32(Subtensor<const float> A);
  Tensor geluFp32(Subtensor<const float> A);

  void randFp32(Tensor *tensor);
  void zerosFp32(Subtensor<float> tensor);
  void print1DFp32(Subtensor<const float> A);
  void printNDFp32(Subtensor<const float> A, int pad_space);
  void printFp32(Subtensor<const float> tensor);
  
  bool allCloseFp32(Subtensor<const float> A, Subtensor<const float> B, float rtol, float atol);
  void catFp32(Subtensor<const float> A, Subtensor<const float> B, int dim, Subtensor<float> C);

  // Copy data from src to tgt. tgt and src should have the same dtype and
  // shape
  void copyFp32(Subtensor<const float> src, Subtensor<float> tgt);

  Tensor lookupFp32(Subtensor<const float> table, Subtensor<const LongType> indices);
  Tensor layerNormFp32(
      Subtensor<const float> input,
      Subtensor<const float> weight,
      Subtensor<const float> bias,
      float eps);
  Tensor causalMaskFp32(int max_len);

  struct GEMMArgs {
    bool transA;
    bool transB;
    int M;
    int N;
    int K;
    int lda;
    int ldb;
    int ldc;
  };

  // generate GEMMArgs from the input tensor A, B and output tensor C. dimensions of A could be
  // greater than 2 (for BMM). throw exception if shape mismatch.
  GEMMArgs generateGemmArgs(const Tensor &A, const Tensor &B, const Tensor &C);

  template<typename T>
  void getSubtensors(Subtensor<T> tensor, int subtensorDim, std::vector<T*>& l);

  template<typename T>
  SubtensorList<T> getVectorList(Subtensor<T> A);
  template<typename T>
  SubtensorList<T> getMatrixList(Subtensor<T> A);


  template<typename T>
  bool isShapeMatch(Subtensor<T> A, Subtensor<T> B);
  template<typename T>
  bool isShapeMatchBroadcastB(Subtensor<T> A, Subtensor<T> B);

  // get shape of BMM output tensor from input A and B.
  std::vector<int> getBmmOutputShape(const Tensor &A, const Tensor &B);
};

// sub-tensor. Stores the shape and data pointer to a sub region of the original
// Tensor. It's faster than Tensor when passing as parameters.
template<typename T>
struct CPUOperators::Subtensor {
  ly::Span<const Shape> shape;
  T *data;

  // get sub-tensor of this Subtensor.
  Subtensor<T> subtensor(int index) {
    return Subtensor<T>{
      this->shape.subspan(1),
      data + index * this->shape[0].stride
    };
  }
  const Subtensor<T> subtensor(int index) const {
    return Subtensor<T>{
      this->shape.subspan(1),
      data + index * this->shape[0].stride
    };
  }

  // get dimension or stride for an axis. NOTE: negative index is NOT supported.
  int dimension(int index) const { return shape[index].shape; }
  int stride(int index) const { return shape[index].stride; }

  // get element from 1D sub-tensor. NOTE: elem() will not check the shape.
  T &elem(int index) {
    return data[index * this->shape[0].stride];
  }
  const T &elem(int index) const {
    return data[index * this->shape[0].stride];
  }

  int64_t getNumVec() const {
    int64_t n = 1;
    for (int i = 0; i < rank() - 1; ++i) {
      n *= shape[i].shape;
    }
    return n;
  }

  // number of element.
  int64_t numel() const {
    int64_t n = 1;
    for (const Shape &s : shape) {
      n *= s.shape;
    }
    return n;
  }

  // get rank.
  int rank() const { return shape.size(); }
};

template<typename T>
class CPUOperators::SubtensorList : public ly::NonCopyable {
 public:
  SubtensorList(ly::Span<const Shape> shape, std::vector<T *> &&l)
      : _shape(shape),
        _list(std::move(l)) {}

  SubtensorList(SubtensorList<T> &&l)
      : _shape(l._shape),
        _list(std::move(l._list)) {}

  ly::Span<const Shape> getShape() const { return _shape; }
  int getSize() const { return _list.size(); }
  ly::Span<T *const> getDataPtrList() const { return ly::makeConstSpan(_list); }

  Subtensor<T> getSubtensor(int index) const {
    Subtensor<T> tensor;
    tensor.data = _list[index];
    tensor.shape = _shape;

    return tensor;
  }

 private:
  ly::Span<const Shape> _shape;
  std::vector<T *> _list;
};

}  // namespace flint

