set(llyn_SOURCES
    "internal/log.cc"
    "internal/sprintf.cc"
    "error.cc"
    "ini_config.cc"
    "path.cc"
    "random.cc"
    "reader.cc"
    "strings.cc")

if(WIN32)
    set(llyn_SOURCES
        ${llyn_SOURCES}
        "path_windows.cc"
        "platform_windows.cc"
        "shared_library_windows.cc")
endif(WIN32)
if(UNIX)
    set(llyn_SOURCES
        ${llyn_SOURCES}
        "path_linux.cc"
        "platform_linux.cc")
endif(UNIX)

add_library(llyn STATIC ${llyn_SOURCES})
target_include_directories(llyn PRIVATE ..)
target_compile_features(llyn PRIVATE cxx_std_14)
if(UNIX)
  target_link_libraries(llyn -ldl)
endif(UNIX)

set(llyn_test_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/strings_test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/path_test.cc" PARENT_SCOPE)
