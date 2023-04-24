#ifndef LLAMA_RUNTIME_NN_H_
#define LLAMA_RUNTIME_NN_H_

#include <stdint.h>
#include "common.h"
#include "status.h"
#include "util.h"

namespace llama {
namespace nn {

class TensorView;
class Function;

// 
class Tensor : private util::NonCopyable {
 public:
  friend class Function;

  typedef int8_t RankType;
  typedef uint8_t FlagType;
  typedef int32_t ShapeType;

  Tensor();
  virtual ~Tensor();

  // copy constructor
  Tensor(Tensor &&tensor);
  Tensor &operator=(Tensor &&);

  // get numebr of dimentsions.
  int rank() const { return rank_; }

  // get the size in dimention `d`. `d` supports positive number
  // (index) and negative number (index from back). Crash if `d` is out of
  // boundary
  int shape(int d) const;

  // get stride for dimension `d`. 
  int stride(int d) const;

  // get number of elements in this tensor.
  int numel() const;

  // return true if this tensor is empty.
  bool empty() const;

  // get data type.
  DType dtype() const { return dtype_; }

  // pointer of data in this tensor
  template <typename T>
  T *data() {
    CHECK(TypeID<T>() == dtype_);
    return reinterpret_cast<T *>(data_);
  }

  template <typename T>
  const T *data() const {
    CHECK(TypeID<T>() == dtype_);
    return reinterpret_cast<T *>(data_);
  }

 protected:
  // maxinum number of rank.
  static constexpr int kMaxRank = 16;

  // rank for empty tensor.
  static constexpr int8_t kEmptyRank = -1;

  // this flag means the Tensor owns its data_ pointer.
  static constexpr uint8_t kFlagOwnData = 1;

  // this flag means the Tensor owns its shape_ pointer.
  static constexpr uint8_t kFlagOwnShape = 2;

  // shape_ has 2 * rank_ items. The first part is shape and the second part is
  // the stride
  ShapeType *shape_;

  void *data_;
  DType dtype_;
  RankType rank_;
  FlagType flag_;

  // dispose all resource in this tensor. Free shape_ if the tensor owns shape_
  // ptr_. Free data_ of the tensor owns data_.
  void Dispose();
};

class TensorView : public Tensor {

};

// context for recurrent neural network or auto-regressive decoder 
class Context {
 public:
  const Tensor &Get(const std::string &name);
  void Set(const std::string &name, const Tensor &value);
}; 

// base class for all nn modules
class Module {
 public:
  // initialize the module from context
  virtual Status InitFromContext(Context *ctx) = 0;
};

// linear layer in the nn.
class Linear : public Module {
 public:
  Linear(int hidden_size);

  // initialize the module from context
  Status InitFromContext(Context *ctx) override;

  // forward input through this module and returns the output
  Tensor Forward(const Tensor &input) const;

 private:
  Tensor w_;
  Tensor b_;
};

class Function {
 public:
  Tensor Lookup(const Tensor &table, const Tensor &indices);
  Tensor LayerNorm(const Tensor &input);

  // Matrix product of two tensors.
  // Args:
  //   A <float>(M, N): matrix A;
  //   B <float>(N, K): matrix B;
  // Returns:
  //   (M, K); A dot B
  Tensor MatMul(const Tensor &A, const Tensor &B);
  Tensor BMM(const Tensor &left, const Tensor &right);

  // Apply softmax on the last dimension of input
  // Args:
  //   input <float>(..., L): input tensor
  // Returns:
  //   <float>(..., L): output tensor 
  Tensor Softmax(const Tensor &input);
  Tensor Add(const Tensor &a, const Tensor &b);
  Tensor Transpose(const Tensor &input, int dim0, int dim1);

  // create a tensor with specified shape and dtype. Data in this tensor is
  // uninitialize.
  Tensor Tensor_(std::initializer_list<int> shape, DType dtype);

  // Returns a tensor filled with random numbers from a uniform distribution on
  // the interval [0, 1) 
  Tensor Rand(std::initializer_list<int> shape, DType dtype);

  // Returns a tensor filled with 0
  Tensor Zeros(std::initializer_list<int> shape, DType dtype);

  // Print the tensor to stdout,
  void Print(const Tensor &tensor);
};

}  // namespace nn
}  // namespace llama

#endif  // LLAMA_RUNTIME_NN_H_
