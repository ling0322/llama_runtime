set(llyn_SOURCES
    "internal/log.cc"
    "internal/sprintf.cc"
    "error.cc"
    "ini_config.cc"
    "path.cc"
    "path_windows.cc"
    "platform_windows.cc"
    "random.cc"
    "reader.cc"
    "shared_library_windows.cc"
    "strings.cc")

add_library(llyn STATIC ${llyn_SOURCES} )
target_include_directories(llyn PRIVATE ..)
target_compile_features(llyn PRIVATE cxx_std_14)

set(llyn_test_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/strings_test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/path_test.cc" PARENT_SCOPE)
