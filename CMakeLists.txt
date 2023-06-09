cmake_minimum_required(VERSION 3.11)
project(llama)

if(WIN32)
  add_definitions( "/D_CRT_SECURE_NO_WARNINGS /wd4267 /DCATCH_AMALGAMATED_CUSTOM_MAIN" )
endif(WIN32)

#find_library(LIBONNXRUNTIME onnxruntime PATHS ${onnxruntime_SOURCE_DIR}/lib)
set(llama_INCDIRS src .)
set(llama_SOURCES
    "src/common.cc"
    "src/environment.cc"
    "src/gemm.cc"
    "src/gemm_kernel.cc"
    "src/gemm_kernel_avx512.cc"
    "src/gemm_kernel_avx2.cc"
    "src/gpt2_model.cc"
    "src/ini_config.cc"
    "src/log.cc"
    "src/nn.cc"
    "src/operators.cc"
    "src/operators_cpu.cc"
    "src/reader.cc"
    "src/shared_library_windows.cc"
    "src/strings.cc"
    "src/tensor.cc"
    "src/tokenizer.cc"
    "src/transformer.cc"
    "src/util.cc"
    "src/util_windows.cc")
add_library(llama SHARED ${llama_SOURCES} )
target_include_directories(llama PRIVATE ${llama_INCDIRS})
target_compile_features(llama PRIVATE cxx_std_14)

set(llama_test_SOURCES
    ${llama_SOURCES}
    "src/gemm_test.cc"
    "src/gpt2_model_test.cc"
    "src/nn_test.cc"
    "src/nn_test_helper.cc"
    "src/operators_test.cc"
    "src/test_helper.cc"
    "src/test_main.cc"
    "src/tensor_test.cc"
    "src/tokenizer_test.cc"
    "src/transformer_test.cc"
    "third_party/catch2/catch_amalgamated.cpp")
add_executable(llama_test ${llama_test_SOURCES})
target_include_directories(llama_test PRIVATE ${llama_INCDIRS})

set(OPENBLAS_INCLUDE_DIR ${OPENBLAS_DIR}/include)
set(OPENBLAS_LIB ${OPENBLAS_DIR}/lib/libopenblas.lib)
set(OPENBLAS_DYNLIB ${OPENBLAS_DIR}/bin/libopenblas.dll)


set(llama_benchmark_SOURCES
    ${llama_SOURCES}
    "src/gemm_benchmark.cc"
    "src/test_helper.cc"
    "src/test_main.cc"
    "third_party/catch2/catch_amalgamated.cpp")
add_executable(llama_benchmark ${llama_benchmark_SOURCES})
target_link_libraries(llama_benchmark ${OPENBLAS_LIB})
target_include_directories(llama_benchmark PRIVATE ${llama_INCDIRS} ${OPENBLAS_INCLUDE_DIR})
add_custom_command(TARGET llama_benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_DYNLIB} $<TARGET_FILE_DIR:llama_benchmark>)
