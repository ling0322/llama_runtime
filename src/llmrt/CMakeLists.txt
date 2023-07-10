set(llmrt_INCDIRS .. ../../third_party)
set(llmrt_SOURCES
    "environment.cc"
    "gpt2_model.cc"
    "transformer.cc")

set(llmrt_LIBADD
    OpenMP::OpenMP_CXX
    llyn
    flint
    gentok
    pmpack)

add_library(llmrt STATIC ${llmrt_SOURCES} )
target_include_directories(llmrt PRIVATE ${llmrt_INCDIRS})
target_compile_features(llmrt PRIVATE cxx_std_14)
target_link_libraries(llmrt ${llmrt_LIBADD})

set(llmrt_test_SOURCES
    "test_main.cc"
    "gpt2_model_test.cc"
    "transformer_test.cc"
    "../../third_party/catch2/catch_amalgamated.cpp")

add_executable(llmrt_test ${llmrt_test_SOURCES})
target_include_directories(llmrt_test PRIVATE ${llmrt_INCDIRS})
target_link_libraries(llmrt_test llmrt)

set(OPENBLAS_INCLUDE_DIR ${OPENBLAS_DIR}/include)
set(OPENBLAS_LIB ${OPENBLAS_DIR}/lib/libopenblas.lib)
set(OPENBLAS_DYNLIB ${OPENBLAS_DIR}/bin/libopenblas.dll)

set(llmrt_benchmark_SOURCES
    "gpt2_model_benchmark.cc"
    "test_main.cc"
    "../../third_party/catch2/catch_amalgamated.cpp")
add_executable(llmrt_benchmark ${llmrt_benchmark_SOURCES})
target_link_libraries(llmrt_benchmark ${OPENBLAS_LIB} OpenMP::OpenMP_CXX)
target_include_directories(llmrt_benchmark PRIVATE ${llmrt_INCDIRS} ${OPENBLAS_INCLUDE_DIR})
add_custom_command(TARGET llmrt_benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_DYNLIB} $<TARGET_FILE_DIR:llmrt_benchmark>)

