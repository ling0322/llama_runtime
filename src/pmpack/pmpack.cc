#include "pmpack/pmpack.h"

#include <stdlib.h>
#include <memory>
#include "pmpack/gemm_fp32qint4fp32.h"
#include "pmpack/sgemm.h"
#include "util/log.h"
#include "util/util.h"

namespace llama {
namespace nn {

enum class CPUMathBackend {
  DEFAULT,
  AVX2,
  AVX512
};

CPUMathBackend findBestCpuMathBackend() {
  if (util::isAvx512Available()) {
    LOG(INFO) << "pmpack: Use Avx512 backend.";
    return CPUMathBackend::AVX512;
  } else if (util::isAvx2Available()) {
    LOG(INFO) << "pmpack: Use Avx2 backend.";
    return CPUMathBackend::AVX2;
  } else {
    LOG(WARN) << "pmpack: fallback to default backend.";
    return CPUMathBackend::AVX2;
  }
}

// instance of PMPack.
class PMPack;
static PMPack *gPmpackInstance = nullptr;

// interface for PMPack.
class PMPack {
 public:
  static void init();
  static void destroy();
  static void setNumThreads(int numThreads);
  static int getNumThreads() { return _numThreads; }
  static const PMPack *getInstance();

  // get kernel implementations.
  const SGEMM *getSgemm() const { return _sgemm.get(); }
  const IGemmFp32QInt4Fp32 *getGemmFp32QInt4Fp32() const { return _gemmFp32QInt4Fp32.get(); }

 private:
  static PMPack *_instance;
  static int _numThreads;

  std::unique_ptr<SGEMM> _sgemm;
  std::unique_ptr<IGemmFp32QInt4Fp32> _gemmFp32QInt4Fp32;
};

PMPack *PMPack::_instance = nullptr;
int PMPack::_numThreads = 1;

void PMPack::init() {
  if (_instance) {
    destroy();
  }

  _instance = new PMPack();
  switch (findBestCpuMathBackend()) {
    case CPUMathBackend::AVX512:
      _instance->_sgemm = std::make_unique<SGEMMImplAvx512>();
      _instance->_gemmFp32QInt4Fp32 = std::make_unique<GemmFp32QInt4Fp32Avx512>();
      break;
    case CPUMathBackend::AVX2:
      _instance->_sgemm = std::make_unique<SGEMMImplAvx2>();
      _instance->_gemmFp32QInt4Fp32 = std::make_unique<GemmFp32QInt4Fp32Avx2>();
      break;
    case CPUMathBackend::DEFAULT:
      _instance->_sgemm = std::make_unique<SGEMMImplDefault>();
      _instance->_gemmFp32QInt4Fp32 = std::make_unique<GemmFp32QInt4Fp32Fallback>();
      break;
    default:
      NOT_IMPL();
  }
}

void PMPack::destroy() {
  delete _instance;
  _instance = nullptr;
}

void PMPack::setNumThreads(int numThreads) {
  _numThreads = numThreads;
}

const PMPack *PMPack::getInstance() {
  CHECK(_instance);
  return _instance;
}

}  // namespace nn
}  // namespace llama

void pmpack_init() {
  llama::nn::PMPack::init();
}

void pmpack_set_num_threads(int32_t num_threads) {
  llama::nn::PMPack::setNumThreads(num_threads);
}

int32_t pmpack_get_num_threads() {
  return llama::nn::PMPack::getNumThreads();
}

void pmpack_destroy() {
  llama::nn::PMPack::destroy();
}

void pmpack_sgemm(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float *C,
    int ldc) {
  llama::nn::PMPack::getInstance()->getSgemm()->apply(
      transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
}

void pmpack_sgemm_batch(
    int batch_size,
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *const *batchA,
    int lda,
    const float *const *batchB,
    int ldb,
    float *const *batchC,
    int ldc) {
  llama::nn::PMPack::getInstance()->getSgemm()->applyBatch(
      batch_size, transA, transB, M, N, K, batchA, lda, batchB, ldb, batchC, ldc);
}

void pmpack_gemm_fp32qint4fp32(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const void *B,
    const float *scaleDataB,
    int groupSizeB,
    float *C,
    int ldc) {
  llama::nn::PMPack::getInstance()->getGemmFp32QInt4Fp32()->apply(
      transA, transB, M, N, K, A, lda, B, scaleDataB, groupSizeB, C, ldc);
}

void pmpack_gemm_fp32qint4fp32_batch(
    int batchSize,
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *const *batchA,
    int lda,
    const void *const *batchB,
    const float *const *batchScaleB,
    int groupSizeB,
    float *const *batchC,
    int ldc) {
  llama::nn::PMPack::getInstance()->getGemmFp32QInt4Fp32()->applyBatch(
      batchSize,
      transA,
      transB,
      M,
      N,
      K,
      batchA,
      lda,
      batchB,
      batchScaleB,
      groupSizeB,
      batchC,
      ldc);
}
