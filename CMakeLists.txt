# project-root/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(llama_go_lib VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the external llama.cpp submodule
# This assumes that llama.cpp is located in the project-root/llama.cpp directory
add_subdirectory(llama.cpp)

# Add your own llama-go.cpp subdirectory
add_subdirectory(llama-go.cpp)

# Optionally, you can define installation rules, packaging, etc., here
