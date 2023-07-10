set(flint_SOURCES
    "cpu_operators.cc"
    "device.cc"
    "dtype.cc"
    "nn.cc"
    "operators.cc"
    "tensor_data.cc"
    "tensor.cc"
    "util.cc")

add_library(flint STATIC ${flint_SOURCES} )
target_include_directories(flint PRIVATE ..)
target_compile_features(flint PRIVATE cxx_std_14)

set(flint_test_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/nn_test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/operators_test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/tensor_test.cc"
    PARENT_SCOPE)
