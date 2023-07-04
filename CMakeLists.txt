cmake_minimum_required(VERSION 3.11)
project(llama)

find_package(OpenMP REQUIRED)

if(WIN32)
  add_definitions( "/D_CRT_SECURE_NO_WARNINGS /wd4267 /DCATCH_AMALGAMATED_CUSTOM_MAIN" )
endif(WIN32)

set_source_files_properties("src/pmpack/gemm_kernel_avx512.cc" PROPERTIES COMPILE_FLAGS /arch:AVX512)
set_source_files_properties("src/pmpack/gemm_kernel_avx2.cc" PROPERTIES COMPILE_FLAGS /arch:AVX2)

set(llama_INCDIRS src .)
set(llama_SOURCES
    "src/common/common.cc"
    "src/common/environment.cc"
    "src/model/gpt2_model.cc"
    "src/model/transformer.cc"
    "src/nn/nn.cc"
    "src/nn/operators.cc"
    "src/nn/operators_cpu.cc"
    "src/nn/tensor.cc"
    "src/pmpack/block.cc"
    "src/pmpack/gemm_kernel_avx512.cc"
    "src/pmpack/gemm_kernel_avx2.cc"
    "src/pmpack/gemm_kernel_fallback.cc"
    "src/pmpack/pmpack.cc"
    "src/pmpack/util.cc"
    "src/tokenizer/tokenizer.cc"
    "src/util/ini_config.cc"
    "src/util/log.cc"
    "src/util/random.cc"
    "src/util/reader.cc"
    "src/util/shared_library_windows.cc"
    "src/util/strings.cc"
    "src/util/util.cc"
    "src/util/util_windows.cc")

add_library(llama SHARED ${llama_SOURCES} )
target_include_directories(llama PRIVATE ${llama_INCDIRS})
target_compile_features(llama PRIVATE cxx_std_14)

set(llama_test_SOURCES
    ${llama_SOURCES}
    "src/common/test_helper.cc"
    "src/common/test_main.cc"
    "src/model/gpt2_model_test.cc"
    "src/model/transformer_test.cc"
    "src/nn/nn_test_helper.cc"
    "src/nn/nn_test.cc"
    "src/nn/operators_test.cc"
    "src/nn/tensor_test.cc"
    "src/pmpack/gemm_test.cc"
    "src/tokenizer/tokenizer_test.cc"
    "third_party/catch2/catch_amalgamated.cpp")
add_executable(llama_test ${llama_test_SOURCES})
target_include_directories(llama_test PRIVATE ${llama_INCDIRS})

set(OPENBLAS_INCLUDE_DIR ${OPENBLAS_DIR}/include)
set(OPENBLAS_LIB ${OPENBLAS_DIR}/lib/libopenblas.lib)
set(OPENBLAS_DYNLIB ${OPENBLAS_DIR}/bin/libopenblas.dll)

set(llama_benchmark_SOURCES
    ${llama_SOURCES}
    "src/model/gpt2_model_benchmark.cc"
    "src/nn/nn_test_helper.cc"
    "src/nn/operators_benchmark.cc"
    "src/common/test_helper.cc"
    "src/common/test_main.cc"
    "third_party/catch2/catch_amalgamated.cpp")
add_executable(llama_benchmark ${llama_benchmark_SOURCES})
target_link_libraries(llama_benchmark ${OPENBLAS_LIB} OpenMP::OpenMP_CXX)
target_include_directories(llama_benchmark PRIVATE ${llama_INCDIRS} ${OPENBLAS_INCLUDE_DIR})
add_custom_command(TARGET llama_benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_DYNLIB} $<TARGET_FILE_DIR:llama_benchmark>)
