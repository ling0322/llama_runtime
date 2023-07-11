

if(WIN32)
    set_source_files_properties("gemm_kernel_avx512.cc" PROPERTIES COMPILE_FLAGS /arch:AVX512)
    set_source_files_properties("gemm_kernel_avx2.cc" PROPERTIES COMPILE_FLAGS /arch:AVX2)
endif(WIN32)
if(UNIX)
    set_source_files_properties("gemm_kernel_avx512.cc" PROPERTIES COMPILE_FLAGS "-mavx512f")
    set_source_files_properties("gemm_kernel_avx2.cc" PROPERTIES COMPILE_FLAGS "-march=native")
endif(UNIX)

set(pmpack_SOURCES
    "block.cc"
    "gemm_kernel_avx512.cc"
    "gemm_kernel_avx2.cc"
    "gemm_kernel_fallback.cc"
    "pmpack.cc"
    "util.cc")

add_library(pmpack STATIC ${pmpack_SOURCES} )
target_include_directories(pmpack PRIVATE ..)
target_compile_features(pmpack PRIVATE cxx_std_14)

set(pmpack_test_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/gemm_test.cc" PARENT_SCOPE)
