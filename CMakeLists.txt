cmake_minimum_required(VERSION 3.14)
project(llama_go_lib VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set BUILD_SHARED_LIBS to OFF to build static libraries
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

# Add the subdirectories
add_subdirectory(llama.cpp)
add_subdirectory(llama-go.cpp)

