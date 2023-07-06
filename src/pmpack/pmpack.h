#pragma once

#include <stdint.h>

void pmpack_init();
void pmpack_set_num_threads(int32_t num_threads);
int32_t pmpack_get_num_threads();
void pmpack_destroy();

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
    int ldc);

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
    int ldc);

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
    int ldc);

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
    int ldc);
