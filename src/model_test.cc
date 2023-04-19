#include <iostream>
#include "common.h"
#include "tensor.h"
#include "test_helper.h"
#include "model.h"
#include "reader.h"

using namespace llama;
using namespace llama::model;

constexpr float kEps = 1e-5f;

bool CheckPort(const Port *port, DType dtype, std::vector<int> shape) {
  if (!port) return false;
  if (port->dtype() != dtype) return false;
  if (port->rank() != shape.size()) return false;
  for (int i = 0; i < shape.size(); ++i) {
    if (port->shape(i) != shape[i]) return false;
  }
  return true;
}

TEST_CASE("onnx model load success", "[core][inferencer]") {
  auto model = Model::FromOnnx("data/test/tiny_mlp.onnx");
  REQUIRE(model.ok());

  REQUIRE(model->num_inputs() == 2);
  REQUIRE(model->num_outputs() == 3);

  // get input output by index
  REQUIRE(model->get_input(0));
  REQUIRE(model->get_input(1));
  REQUIRE_FALSE(model->get_input(2));
  REQUIRE(model->get_output(0));
  REQUIRE(model->get_output(1));
  REQUIRE(model->get_output(2));
  REQUIRE_FALSE(model->get_output(3));

  // get input output by name
  REQUIRE(model->get_input("a"));
  REQUIRE(model->get_input("b"));
  REQUIRE_FALSE(model->get_input("c"));
  REQUIRE(model->get_output("c"));
  REQUIRE(model->get_output("d"));
  REQUIRE(model->get_output("e"));
  REQUIRE_FALSE(model->get_output("a"));

  // check shape
  REQUIRE(CheckPort(model->get_input("a"), DType::kFloat, {-1, 16}));
  REQUIRE(CheckPort(model->get_input("b"), DType::kFloat, {-1, 16}));
  REQUIRE(CheckPort(model->get_output("c"), DType::kFloat, {-1, 16}));
  REQUIRE(CheckPort(model->get_output("d"), DType::kFloat, {-1, 16}));
  REQUIRE(CheckPort(model->get_output("e"), DType::kFloat, {-1, 16}));
}

Status RunOneTest(const Model *model, ReadableFile *fp) {
  Tensor a, b, c_ref, d_ref, e_ref;
  RETURN_IF_ERROR(a.Read(fp));
  RETURN_IF_ERROR(b.Read(fp));
  RETURN_IF_ERROR(c_ref.Read(fp));
  RETURN_IF_ERROR(d_ref.Read(fp));
  RETURN_IF_ERROR(e_ref.Read(fp));

  auto infer_request = model->CreateInferRequest();
  Value c, d, e;
  infer_request->SetInput("a", a.View());
  infer_request->SetInput("b", b.View());
  infer_request->SetOutput("c", &c);
  infer_request->SetOutput("d", &d);
  infer_request->SetOutput("e", &e);
  RETURN_IF_ERROR(infer_request->Infer());

  TensorView cv, dv, ev;
  RETURN_IF_ERROR(c.GetTensor(&cv));
  RETURN_IF_ERROR(d.GetTensor(&dv));
  RETURN_IF_ERROR(e.GetTensor(&ev));

  REQUIRE(test_helper::IsNear(c_ref.View(), cv, kEps));
  REQUIRE(test_helper::IsNear(d_ref.View(), dv, kEps));
  REQUIRE(test_helper::IsNear(e_ref.View(), ev, kEps));

  return OkStatus();
}

TEST_CASE("ORT model inference works", "[core][model]") {
  auto model = Model::FromOnnx("data/test/tiny_mlp.onnx");
  REQUIRE(model.ok());

  auto fp = ReadableFile::Open("data/test/tiny_mlp_tensors.bin");
  REQUIRE(fp.ok());

  Status status = RunOneTest(model.get(), fp.get());
  REQUIRE(status.ok());

  status = RunOneTest(model.get(), fp.get());
  REQUIRE(status.ok());
}
