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

