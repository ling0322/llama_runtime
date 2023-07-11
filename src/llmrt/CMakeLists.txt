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

add_executable(llmrt_test
    ${flint_test_SOURCES}
    ${gentok_test_SOURCES}
    ${llmrt_test_SOURCES}
    ${llyn_test_SOURCES}
    ${pmpack_test_SOURCES})
target_include_directories(llmrt_test PRIVATE ${llmrt_INCDIRS})
target_link_libraries(llmrt_test llmrt)
