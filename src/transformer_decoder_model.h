#ifndef LLAMA_CC_TRANSFORMER_DECODER_MODEL_H_
#define LLAMA_CC_TRANSFORMER_DECODER_MODEL_H_

#include "status.h"
#include "tensor.h"
#include "util.h"

namespace llama {

class InferenceEngine;
class Value;

// seq2seq decoder model (with past state). Support auto regressive inferencing
class TransformerDecoderModel {
 public:
  // load the Whisper decoder model from 
  static StatusOr<TransformerDecoderModel> FromOnnx(
      const std::string &model_filename);

  TransformerDecoderModel(TransformerDecoderModel &) = delete;
  TransformerDecoderModel &operator=(TransformerDecoderModel &) = delete;

  Status Forward(const TensorView1Dl &input_ids,
                 const TensorView1Dl &source_indices,
                 std::vector<Value> *past, Value *logits) const;

 private:
  std::unique_ptr<InferenceEngine> inference_engine_;
  std::vector<std::string> past_inputs_;
  std::vector<std::string> present_outputs_;

  TransformerDecoderModel();

  void GetPastInputs();
  void GetPresentOutputs();
  Status CheckPorts();

};

}  // namespace llama

#endif  // LLAMA_CC_TRANSFORMER_DECODER_MODEL_H_
