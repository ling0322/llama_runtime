#ifndef LLAMA_CC_MODEL_H_
#define LLAMA_CC_MODEL_H_

#include <stdint.h>
#include <memory>
#include <vector>

#include "tensor.h"

namespace llama {
namespace model {

class Port;
class InferRequest;

// A neural network model inference engine. Use CreateInferRequest() to infer.
// 
// Example:
//
// StatusOr<Model> model = Model::FromOnnx(onnx_file);
// if (!model.ok()) { ... }
//
// TensorView a = GetTensorXXX();
// TensorView b = GetTensorXXX();
// Value c_v;
//
// std::unique_ptr<InferRequest> infer_request = model->CreateInferRequest();
// infer_request->SetInput("A", a);
// infer_request->SetInput("B", b);
// infer_request->SetOutput("C", &c_v);
//
// RETURN_IF_ERROR(infer_request->Infer());
//
// TensorView c;
// RETURN_IF_ERROR(c_v.GetTensor(&c));
//
class Model {
 public:
  virtual ~Model() = default;

  // initialize the model from onnx model
  static StatusOr<Model> FromOnnx(const std::string &onnx_model_file);

  // get input and output port, return nullptr if name not exist.
  const Port *get_input(const std::string &name) const noexcept;
  const Port *get_output(const std::string& name) const noexcept;
  const Port *get_input(int index) const noexcept;
  const Port *get_output(int index) const noexcept;
  int num_inputs() const noexcept;
  int num_outputs() const noexcept;

  // run the inferencing with given context
  virtual std::unique_ptr<InferRequest> CreateInferRequest() const = 0;

 protected:
  virtual const std::vector<Port> &Inputs() const = 0;
  virtual const std::vector<Port> &Outputs() const = 0;

  Model() = default;
};

// Shape and dtype information of the input and output value
class Port {
 public:
  enum class Type {
    kUnknown,
    kInput,
    kOutput
  };

  Port();
  Port(Port &&) = default;

  // type
  Type type() const;
  void set_type(Type type);

  // name
  PCStrType name() const;
  void set_name(const std::string &name);

  // dtype
  DType dtype() const;
  void set_dtype(DType dtype);

  // shape
  int rank() const;
  int shape(int d) const;
  void set_shape(const std::vector<int> &shape);

 private:
  Type type_;
  DType dtype_;
  std::vector<int> shape_;
  std::string name_;
};

// type of backend runtime providers
enum class BackendType {
  kUnknown,
  kORT
};

// value used in the input and output of inferencing
class Value {
 public:
  class ImplBase;

  Value() = default;
  Value(std::unique_ptr<ImplBase> &&impl);
  Value(Value &&value) noexcept = default;
  Value(Value &value) = delete;
  Value &operator=(Value &) = delete;
  Value &operator=(Value &&) = default;

  Status GetTensor(TensorView *tensor);
  ImplBase *impl() { return impl_.get(); }
  const ImplBase *impl() const { return impl_.get(); }

 private:
  std::unique_ptr<ImplBase> impl_;
};

class Value::ImplBase {
 public:
  virtual ~ImplBase() = default;

  virtual Status GetTensor(TensorView *tensor) = 0;
  virtual BackendType backend_type() const = 0;
};


// Interface for InferRequest
class InferRequest {
 public:
  virtual void SetInput(PCStrType name, const Value &value) = 0;
  virtual void SetOutput(PCStrType name, Value *value) = 0;
  virtual Status Infer() = 0;

  void SetInput(PCStrType name, const TensorView &tensor);

 protected:
  std::vector<Value> borrowed_values_;
  virtual Value CreateValue(const TensorView &tensor) = 0;

  InferRequest() = default;
};

}  // namespace model
}  // namespace llama

#endif  // LLAMA_CC_MODEL_H_
