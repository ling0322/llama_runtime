cmake_minimum_required(VERSION 3.11)
project(llama)

find_package(OpenMP REQUIRED)

if(WIN32)
  add_definitions( "/D_CRT_SECURE_NO_WARNINGS /wd4267 /DCATCH_AMALGAMATED_CUSTOM_MAIN" )
endif(WIN32)

add_subdirectory("src/llyn")
add_subdirectory("src/flint")
add_subdirectory("src/gentok")
add_subdirectory("src/pmpack")
add_subdirectory("src/llmrt")
