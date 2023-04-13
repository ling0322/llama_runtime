#ifndef LLAMA_CC_ONNXRUNTIME_MODEL_H_
#define LLAMA_CC_ONNXRUNTIME_MODEL_H_

#include <stdint.h>
#include <memory>
#include <vector>

#include "model.h"
#include "span.h"
#include "tensor.h"
#include "util.h"

enum ONNXTensorElementDataType;
struct OrtApi;
struct OrtEnv;
struct OrtMemoryInfo;
struct OrtSession;
struct OrtStatus;
struct OrtTypeInfo;
struct OrtValue;
struct OrtTensorTypeAndShapeInfo;

namespace llama {
namespace model {

// defined later
class ORTHelper;

// Inferencer woth onnxruntime backend
class ORTModel : public Model,
                 public NonCopyable {
 public:
  ORTModel();
  ~ORTModel();

  Status Init(const std::string &onnx_model_path);

  const std::vector<Port> &Inputs() const override;
  const std::vector<Port> &Outputs() const override;

  std::unique_ptr<InferRequest> CreateInferRequest() const override;

 private:
  typedef const OrtValue *PCOrtValue;
  typedef OrtValue *POrtValue;
  
  std::unique_ptr<ORTHelper> helper_;
  AutoCPtr<OrtSession> session_;
  AutoCPtr<OrtMemoryInfo> cpu_memory_info_;

  // input and output value information
  std::vector<Port> inputs_;
  std::vector<Port> outputs_;
};


// helper class for OrtApi from onnxruntime
class ORTHelper {
 public:
  ORTHelper();
  ORTHelper(const OrtApi *api);
  ~ORTHelper();

  // return the api pointer
  const OrtApi *api() const { return api_; }

  // Create a ORT session from model data
  Status CreateSession(
      const OrtEnv *ort_env,
      const Span<CByteType> model_data,
      AutoCPtr<OrtSession> *session);

  // create an `OrtValue` whose shape and data are copied from `tensor_view`.
  // The returned OrtValue will own the data.
  AutoCPtr<OrtValue> CreateValue(const TensorView &tensor_view);

  // create an OrtValue from TensorView. The OrtValue will borrow the data from
  // TensorView
  AutoCPtr<OrtValue> CreateBorrowedValue(
      const OrtMemoryInfo *memory_info,
      const TensorView &tensor_view);

  // create memory info for CPU
  AutoCPtr<OrtMemoryInfo> CreateCPUMemoryInfo();

  // get information for inputs and outputs of the model, then save them to
  // ports
  Status GetPortList(const OrtSession *session, Port::Type type,
                     std::vector<Port> *ports);

  // convert OrtStatus to Status, then free it
  Status Check(OrtStatus *status);

  // get type and shape info from OrtTensorTypeAndShapeInfo
  Status GetTypeAndShapeInfo(
      const OrtTensorTypeAndShapeInfo *info,
      DType *dtype,
      int *rank,
      std::array<int, TensorView::kMaxRank> *shape);

  // convertion between ONNXTensorElementDataType and DType 
  Status OnnxTypeToDType(
      ONNXTensorElementDataType onnx_dtype,
      DType *dtype);
  Status DtypeToOnnxType(
      DType dtype,
      ONNXTensorElementDataType *onnx_dtype);


 private:
  // pointer to OrtApi, from Env::instance().ort_env(), do not need to
  // release manually
  const OrtApi *api_;

  // get Inferencer::Info from OrtTypeInfo
  Status GetInfoFromOrtTypeInfo(const OrtTypeInfo *type_info, Port &port);

  // convert shape from `tensor` to int64_t array `shape`
  void CvtShapeToInt64(
      const TensorView &tensor,
      std::array<int64_t, TensorView::kMaxRank> &shape);
};


// Interface for InferRequest
class ORTInferRequest : public InferRequest {
 public:
  ORTInferRequest(const OrtApi *api,
                  const OrtMemoryInfo *memory_info,
                  OrtSession *session);

  void SetInput(PCStrType name, const Value &value) override;
  void SetOutput(PCStrType name, Value *value) override;

  Status Infer() override;

 protected:
  std::vector<const OrtValue *> inputs_;
  std::vector<std::string> input_names_;

  std::vector<Value *> outputs_;
  std::vector<std::string> output_names_;

  const OrtApi *api_;
  const OrtMemoryInfo *memory_info_;
  OrtSession *session_;
  ORTHelper ort_helper_;

  Value CreateValue(const TensorView &tensor) override;
};

// Wrapper for OrtValue
class ORTValue : public Value::ImplBase, public NonCopyable {
 public:
  // create the ORTInferValue from OrtValue
  ORTValue();
  ORTValue(const OrtApi *api, OrtValue *value);
  ~ORTValue();

  // implements interface Value::ImplBase
  Status GetTensor(TensorView *tensor) override;
  BackendType backend_type() const override;

  const OrtValue *value() const;

 private:
  AutoCPtr<OrtValue> value_;
  ORTHelper helper_;
};

}  // namespace model
}  // namespace llama

#endif  // LLAMA_CC_ONNXRUNTIME_MODEL_H_
