#include "onnxruntime_model.h"

#include "env.h"
#include "readable_file.h"
#include "third_party/onnxruntime/onnxruntime_c_api.h"

namespace llama {
namespace model {

// ----------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------

StatusOr<Model> Model::FromOnnx(
    const std::string &onnx_model_file) {
  auto model = std::make_unique<ORTModel>();
  RETURN_IF_ERROR(model->Init(onnx_model_file));

  return std::move(model);
}

// ----------------------------------------------------------------------------
// ORTModel
// ----------------------------------------------------------------------------

ORTModel::ORTModel() {}
ORTModel::~ORTModel() {}

Status ORTModel::Init(const std::string &onnx_model_path) {
  const Env *env = Env::instance();  
  const OrtApi *ort_api = env->ort_api();
  OrtEnv *ort_env = env->ort_env();
  if (!ort_env) {
    RETURN_ABORTED() << "onnxruntime not enabled. Please check the error log.";
  }

  helper_ = std::make_unique<ORTHelper>(ort_api);

  // create memory info
  cpu_memory_info_ = helper_->CreateCPUMemoryInfo();

  // read model
  StatusOr<ReadableFile> fp = ReadableFile::Open(onnx_model_path);

  std::vector<ByteType> model_data;
  RETURN_IF_ERROR(fp->ReadAll(&model_data));
  RETURN_IF_ERROR(helper_->CreateSession(
      ort_env,
      MakeConstSpan(model_data),
      &session_));
  RETURN_IF_ERROR(helper_->GetPortList(
      session_.get(),
      Port::Type::kInput,
      &inputs_));
  RETURN_IF_ERROR(helper_->GetPortList(
      session_.get(),
      Port::Type::kOutput,
      &outputs_));

  return OkStatus();
}

const std::vector<Port> &ORTModel::Inputs() const {
  return inputs_;
}
const std::vector<Port> &ORTModel::Outputs() const {
  return outputs_;
}

std::unique_ptr<InferRequest> ORTModel::CreateInferRequest() const {
  return std::make_unique<ORTInferRequest>(
      helper_->api(),
      cpu_memory_info_.get(),
      const_cast<OrtSession *>(session_.get()));
}

// ----------------------------------------------------------------------------
// ORTHelper
// ----------------------------------------------------------------------------

ORTHelper::ORTHelper(): api_(nullptr) {}
ORTHelper::ORTHelper(const OrtApi *api): api_(api) {}
ORTHelper::~ORTHelper() {
  api_ = nullptr;
}

Status ORTHelper::Check(OrtStatus *status) {
  if (!status) {
    return OkStatus();
  } else {
    AutoCPtr<OrtStatus> ort_status(status, api_->ReleaseStatus);
    RETURN_ABORTED() << api_->GetErrorMessage(ort_status.get());
  }
}

Status ORTHelper::CreateSession(
    const OrtEnv *ort_env,
    Span<CByteType> model_data,
    AutoCPtr<OrtSession> *session) {
  // create ORT session
  AutoCPtr<OrtSessionOptions> options(nullptr, api_->ReleaseSessionOptions);
  RETURN_IF_ERROR(Check(api_->CreateSessionOptions(options.get_pp())));

  // set options
  RETURN_IF_ERROR(Check(api_->SetIntraOpNumThreads(options.get(), 1)));
  RETURN_IF_ERROR(Check(api_->SetSessionGraphOptimizationLevel(
      options.get(),
      ORT_ENABLE_BASIC)));

  *session = AutoCPtr<OrtSession>(nullptr, api_->ReleaseSession);
  RETURN_IF_ERROR(Check(api_->CreateSessionFromArray(
      ort_env,
      model_data.data(),
      model_data.size(),
      options.get(),
      session->get_pp())));;
  
  return OkStatus();
}

AutoCPtr<OrtValue> ORTHelper::CreateValue(const TensorView &tensor_view) {
  // should not release alloc
  OrtAllocator *alloc = nullptr;
  LL_CHECK_OK(Check(api_->GetAllocatorWithDefaultOptions(&alloc)));

  // prepare shape
  std::array<int64_t, TensorView::kMaxRank> shape;
  CvtShapeToInt64(tensor_view, shape);

  AutoCPtr<OrtValue> ort_value = {nullptr, api_->ReleaseValue};
  ONNXTensorElementDataType onnx_dtype;
  LL_CHECK_OK(DtypeToOnnxType(tensor_view.dtype(), &onnx_dtype));

  LL_CHECK_OK(Check(api_->CreateTensorAsOrtValue(
      alloc,
      shape.data(),
      tensor_view.rank(),
      onnx_dtype,
      ort_value.get_pp())));

  // copy data from `tensor_view` to `ort_value`
  void *data = nullptr;
  LL_CHECK_OK(Check(api_->GetTensorMutableData(
      ort_value.get(), 
      &data)));
  int nb = tensor_view.numel() * SizeOfDType(tensor_view.dtype());
  memcpy(data, tensor_view.raw_data(), nb);

  return ort_value;
}

AutoCPtr<OrtValue> ORTHelper::CreateBorrowedValue(
    const OrtMemoryInfo *memory_info,
    const TensorView &tensor) {
  // prepare shape
  std::array<int64_t, TensorView::kMaxRank> shape;
  CvtShapeToInt64(tensor, shape);

  // create value
  AutoCPtr<OrtValue> ort_value = {nullptr, api_->ReleaseValue};

  ONNXTensorElementDataType onnx_dtype;
  LL_CHECK_OK(DtypeToOnnxType(tensor.dtype(), &onnx_dtype));
  LL_CHECK_OK(Check(api_->CreateTensorWithDataAsOrtValue(
      memory_info,
      const_cast<void *>(tensor.raw_data()),
      tensor.numel() * SizeOfDType(tensor.dtype()),
      shape.data(),
      tensor.rank(),
      onnx_dtype,
      ort_value.get_pp())));

  return ort_value;
}

Status ORTHelper::GetPortList(const OrtSession *session,
                              Port::Type type,
                              std::vector<Port> *ports) {
  // should not release alloc
  OrtAllocator *alloc = nullptr;
  RETURN_IF_ERROR(Check(api_->GetAllocatorWithDefaultOptions(&alloc)));

  auto count_func = api_->SessionGetInputCount;
  auto name_func = api_->SessionGetInputName;
  auto type_info_func = api_->SessionGetInputTypeInfo;
  switch (type) {
    case Port::Type::kInput:
      break;
    case Port::Type::kOutput:
      count_func = api_->SessionGetOutputCount;
      name_func = api_->SessionGetOutputName;
      type_info_func = api_->SessionGetOutputTypeInfo;
      break;
    default:
      NOT_IMPL();
  }

  // get number of inputs or outputs
  size_t n_inputs = 0;
  RETURN_IF_ERROR(Check(count_func(session, &n_inputs)));

  // find the input/output index by name
  ports->clear();
  for (size_t i = 0; i < n_inputs; ++i) {
    Port port;
    port.set_type(type);

    // name
    AutoCPtr<char> name(nullptr, std::bind(
        alloc->Free, alloc, std::placeholders::_1));
    RETURN_IF_ERROR(Check(name_func(session, i, alloc, name.get_pp())));
    port.set_name(name.get());

    // type_info
    AutoCPtr<OrtTypeInfo> type_info(nullptr, api_->ReleaseTypeInfo);
    RETURN_IF_ERROR(Check(type_info_func(session, i, type_info.get_pp())));

    // extract dtype and shape from type_info
    RETURN_IF_ERROR(GetInfoFromOrtTypeInfo(type_info.get(), port));

    ports->emplace_back(std::move(port));
  }

  return OkStatus();
}

Status ORTHelper::GetTypeAndShapeInfo(
    const OrtTensorTypeAndShapeInfo *info,
    DType *dtype,
    int *rank,
    std::array<int, TensorView::kMaxRank> *shape) {
  // shape
  size_t rank_sizet = 0;
  RETURN_IF_ERROR(Check(api_->GetDimensionsCount(info, &rank_sizet)));
  if (rank_sizet > TensorView::kMaxRank) {
    RETURN_ABORTED() << "too many dimensions in OrtTensor";
  }
  *rank = rank_sizet;

  std::vector<int64_t> ll_shape(rank_sizet);
  RETURN_IF_ERROR(Check(api_->GetDimensions(
      info,
      ll_shape.data(),
      ll_shape.size())));

  std::transform(ll_shape.begin(),
                 ll_shape.end(),
                 shape->begin(),
                 [](int64_t v) { return static_cast<int>(v); });

  // dtype
  *dtype = DType::kUnknown;
  ONNXTensorElementDataType ort_dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  RETURN_IF_ERROR(Check(api_->GetTensorElementType(info, &ort_dtype)));
  RETURN_IF_ERROR(OnnxTypeToDType(ort_dtype, dtype));

  return OkStatus();
}

Status ORTHelper::GetInfoFromOrtTypeInfo(const OrtTypeInfo *type_info,
                                         Port &port) {
  // cast fron TypeInfo to TensorInfo, we do not need to free tensor_info
  const OrtTensorTypeAndShapeInfo *tensor_info = nullptr;
  RETURN_IF_ERROR(Check(api_->CastTypeInfoToTensorInfo(
      type_info,
      &tensor_info)));

  // get dtype and shape from tensor_info
  DType dtype = DType::kUnknown;
  int rank = 0;
  std::array<int, TensorView::kMaxRank> shape;
  RETURN_IF_ERROR(GetTypeAndShapeInfo(tensor_info, &dtype, &rank, &shape));

  // set ValueInfo
  port.set_shape(std::vector<int>(shape.begin(), shape.begin() + rank));
  port.set_dtype(dtype);

  return OkStatus();
}

Status ORTHelper::OnnxTypeToDType(
    ONNXTensorElementDataType onnx_dtype,
    DType *dtype) {
  switch (onnx_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *dtype = DType::kFloat;
      break;
    default:
      RETURN_ABORTED() << "unexpected ONNX dtype: " << onnx_dtype;
  }

  return OkStatus();
}

Status ORTHelper::DtypeToOnnxType(
    DType dtype,
    ONNXTensorElementDataType *onnx_dtype) {
  switch (dtype) {
    case DType::kFloat:
      *onnx_dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      break;
    default:
        RETURN_ABORTED() << "unexpected dtype";
  }

  return OkStatus();
}

AutoCPtr<OrtMemoryInfo> ORTHelper::CreateCPUMemoryInfo() {
  AutoCPtr<OrtMemoryInfo> memory_info = {nullptr, api_->ReleaseMemoryInfo};
  LL_CHECK_OK(Check(api_->CreateCpuMemoryInfo(
      OrtArenaAllocator,
      OrtMemTypeDefault,
      memory_info.get_pp())));

  return memory_info;
}

void ORTHelper::CvtShapeToInt64(
    const TensorView &tensor,
    std::array<int64_t, TensorView::kMaxRank> &shape) {
  LL_CHECK(tensor.rank() <= TensorView::kMaxRank);
  std::transform(tensor.shape_data(),
                 tensor.shape_data() + tensor.rank(),
                 shape.begin(),
                 [](int32_t v) { return static_cast<int64_t>(v); });
}

// ----------------------------------------------------------------------------
// ORTInferRequest
// ----------------------------------------------------------------------------

ORTInferRequest::ORTInferRequest(
    const OrtApi *api,
    const OrtMemoryInfo *memory_info,
    OrtSession *session)
      : api_(api),
        memory_info_(memory_info),
        session_(session),
        ort_helper_(api) {}

void ORTInferRequest::SetInput(PCStrType name, const Value &value) {
  input_names_.push_back(std::string(name));

  LL_CHECK(value.impl()->backend_type() == BackendType::kORT);
  inputs_.push_back(reinterpret_cast<const ORTValue *>(value.impl())->value());
}

void ORTInferRequest::SetOutput(PCStrType name, Value *value) {
  outputs_.push_back(value);
  output_names_.push_back(std::string(name));
}

Status ORTInferRequest::Infer() {
  std::vector<PCStrType> input_names(input_names_.size());
  for (int i = 0; i < input_names_.size(); ++i) {
    input_names[i] = input_names_[i].c_str();
  }

  std::vector<PCStrType> output_names(output_names_.size());
  std::vector<OrtValue *> output_values(outputs_.size());
  for (int i = 0; i < output_names.size(); ++i) {
    output_names[i] = output_names_[i].c_str();
    output_values[i] = nullptr;
  }

  // inferencing
  RETURN_IF_ERROR(ort_helper_.Check(api_->Run(
      const_cast<OrtSession *>(session_),
      nullptr,
      input_names.data(),
      inputs_.data(),
      input_names.size(),
      output_names.data(),
      output_names.size(),
      output_values.data())));
  
  for (int i = 0; i < outputs_.size(); ++i) {
    OrtValue *value = output_values[i];
    *(outputs_[i]) = Value(std::make_unique<ORTValue>(api_, value));
  }

  return OkStatus();
}

Value ORTInferRequest::CreateValue(const TensorView &tensor) {
  AutoCPtr<OrtValue> ort_value = ort_helper_.CreateBorrowedValue(
      memory_info_, tensor);
  return std::make_unique<ORTValue>(api_, ort_value.Release());
}

// ----------------------------------------------------------------------------
// ORTValue
// ----------------------------------------------------------------------------

ORTValue::ORTValue(const OrtApi *api, OrtValue *value):
    value_(value, api->ReleaseValue),
    helper_(api) {
  LL_CHECK(value) << "value is nullptr";
}

BackendType ORTValue::backend_type() const {
  return BackendType::kORT;
}

ORTValue::~ORTValue() {}

Status ORTValue::GetTensor(TensorView *tensor) {
  AutoCPtr<OrtTensorTypeAndShapeInfo> value_info(
      nullptr, helper_.api()->ReleaseTensorTypeAndShapeInfo);
  RETURN_IF_ERROR(helper_.Check(helper_.api()->GetTensorTypeAndShape(
      value_.get(),
      value_info.get_pp())));

  // get dtype and shape  
  std::array<int, TensorView::kMaxRank> shape;
  int rank = 0;
  DType dtype = DType::kUnknown;
  RETURN_IF_ERROR(helper_.GetTypeAndShapeInfo(
      value_info.get(),
      &dtype,
      &rank,
      &shape));

  // get data
  void *data = nullptr;
  RETURN_IF_ERROR(helper_.Check(helper_.api()->GetTensorMutableData(
      const_cast<OrtValue *>(value_.get()), 
      &data)));

  *tensor = TensorView::From(shape.data(), rank, dtype, data);
  return OkStatus();
}

const OrtValue *ORTValue::value() const {
  return value_.get();
}

}  // namespace model
}  // namespace llama
