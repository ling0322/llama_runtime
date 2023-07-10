add_library(gentok STATIC
    "tokenizer.cc")
target_include_directories(gentok PRIVATE ..)
target_compile_features(gentok PRIVATE cxx_std_14)

set(gentok_test_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/tokenizer_test.cc" PARENT_SCOPE)
