#ifndef LLAMA_CC_PROCESSOR_H_
#define LLAMA_CC_PROCESSOR_H_

#include "span.h"

namespace llama {

template<class TIn, class TOut>
class IProcessor {
 public:
  virtual ~IProcessor() = default;

  virtual Status Process(util::Span<const TIn> inputs) = 0;
  virtual Status EndOfStream() = 0;

  virtual std::vector<TOut> GetOutput() = 0;
};

}  // namespace llama

#endif  // LLAMA_CC_PROCESSOR_H_
