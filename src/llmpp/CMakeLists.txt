set(llmpp_INCDIRS .. ../../third_party)
set(llmpp_SOURCES
    "chatglm2_model.cc"
    "environment.cc"
    "gpt2_model.cc"
    "transformer.cc")

set(llmpp_LIBADD
    OpenMP::OpenMP_CXX
    llyn
    flint
    gentok
    pmpack)

add_library(llmpp STATIC ${llmpp_SOURCES} )
target_include_directories(llmpp PRIVATE ${llmpp_INCDIRS})
target_compile_features(llmpp PRIVATE cxx_std_14)
target_link_libraries(llmpp ${llmpp_LIBADD})

set(llmpp_test_SOURCES
    "chatglm2_model_test.cc"
    "test_main.cc"
    "gpt2_model_test.cc"
    "transformer_test.cc"
    "../../third_party/catch2/catch_amalgamated.cpp")

add_executable(llmpp_test
    ${flint_test_SOURCES}
    ${gentok_test_SOURCES}
    ${llmpp_test_SOURCES}
    ${llyn_test_SOURCES}
    ${pmpack_test_SOURCES})
target_include_directories(llmpp_test PRIVATE ${llmpp_INCDIRS})
target_link_libraries(llmpp_test llmpp)
