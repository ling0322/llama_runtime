#include "flint/embedding.h"
#include "flint/nn.h"
#include "llyn/ini_config.h"

namespace llmpp {

struct ChatGLM2Config {
  // config section in ini
  static constexpr char kSection[] = "chatglm2";

  int nEmbd;
  int vocabSize;

  ChatGLM2Config();
  static std::unique_ptr<ChatGLM2Config> fromIni(const ly::IniConfig &ini);
};

// The ChatGLM2 model.
class ChatGLM2Model : public flint::Module {
 public:
  // create BloomModel.
  static std::unique_ptr<ChatGLM2Model> create(const flint::Context &ctx, ChatGLM2Config config);

  // initialize the module from context
  void initParameters(const flint::TensorMap &stateDict) override;


  flint::Tensor forward(flint::TensorMap *past, const flint::Tensor &input) const;

 private:
  flint::Context _ctx;
  ChatGLM2Config _config;

  static constexpr char kChatGlm2[] = "chatglm2";
  static constexpr char kEmbd[] = "embd";

  std::unique_ptr<flint::Embedding> _embedding;

  ChatGLM2Model();
};

}  // namespace llmpp
