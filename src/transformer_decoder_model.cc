#include "transformer_decoder_model.h"

#include "common.h"
#include "inference_engine.h"
#include "strings.h"

namespace llama {

StatusOrPtr<TransformerDecoderModel> TransformerDecoderModel::FromOnnx(
    const std::string &model_filename) {
  auto inference_engine = InferenceEngine::FromOnnx(model_filename);
  if (!inference_engine.ok()) {
    return std::move(inference_engine).status();
  }

  std::unique_ptr<TransformerDecoderModel> model(new TransformerDecoderModel());
  model->inference_engine_ = std::move(inference_engine).pointer();

  model->GetPastInputs();
  model->GetPresentOutputs();
  RETURN_IF_ERROR(model->CheckPorts()) << model_filename;

  return OkStatus();
}

void TransformerDecoderModel::GetPastInputs() {
  int n_inputs = 0;
  for (int n_inputs = 0;; ++n_inputs) {
    std::string name = Sprintf("past_%d", n_inputs);
    if (!inference_engine_->get_input(name)) {
      break;
    }
    past_inputs_.emplace_back(name);
  }
}

void TransformerDecoderModel::GetPresentOutputs() {
  int n_outputs = 0;
  for (int n_outputs = 0;; ++n_outputs) {
    std::string name = Sprintf("present_%d", n_outputs);
    if (!inference_engine_->get_output(name)) {
      break;
    }
    present_outputs_.emplace_back(name);
  }
}

Status TransformerDecoderModel::CheckPorts() {
  if (!inference_engine_->get_input("input_ids")) {
    RETURN_ABORTED() << "input 'input_ids' not found";
  }

  if (!inference_engine_->get_output("source_indices")) {
    RETURN_ABORTED() << "input 'source_indices' not found";
  }

  if (!inference_engine_->get_output("logits")) {
    RETURN_ABORTED() << "output 'logits' not found";
  }

  if (past_inputs_.empty()) {
    RETURN_ABORTED() << "no past_* inputs";
  }

  if (present_outputs_.empty()) {
    RETURN_ABORTED() << "no present_* outputs";
  }

  if (past_inputs_.size() != present_outputs_.size()) {
    RETURN_ABORTED() << "past and present number mismatch";
  }

  return OkStatus();
}

Status TransformerDecoderModel::Forward(
    const TensorView1Dl &input_ids,
    const TensorView1Dl &source_indices,
    std::vector<Value> *past,
    Value *logits) const {
  if (past->size() != past_inputs_.size())
    RETURN_ABORTED() << "invalid past size";

  std::unique_ptr<InferRequest> req = inference_engine_->CreateInferRequest();
  req->SetInput("input_ids", input_ids.ToTensorView());
  req->SetInput("source_indices", source_indices.ToTensorView());
  req->SetOutput("logits", logits);

  std::vector<Value> present(past_inputs_.size());
  for (int i = 0; i < past_inputs_.size(); ++i) {
    req->SetInput(past_inputs_[i].c_str(), (*past)[i]);
    req->SetOutput(present_outputs_[i].c_str(), &(present[i]));
  }

  RETURN_IF_ERROR(req->Infer()) << "decoder-model inference failed";

  *past = std::move(present);
  return OkStatus();
}

}  // namespace llama
