# Go Wrapper for llama.cpp Library

## Precompiled binaries

Precompiled binaries for Windows and Linux are available in the [dist](dist) directory. If the architecture/platform you are using is not available, you would need to compile the library yourself.

## Compile library

First, clone the repository and its submodules with the following commands. The `--recurse-submodules` flag is used to clone the `ggml` submodule, which is a header-only library for matrix operations.

```bash
git clone --recurse-submodules https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### Compile on Linux

Make sure you have a C/C++ compiler and CMake installed. For Ubuntu, you can install them with the following commands:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake
```

Then you can compile the library with the following commands:

```bash
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
make all
```

This should generate `libllama.so` and `libggml.so` that you can use. You can also install the library by coping it into `/usr/lib`.

### Compile on Windows

Make sure you have a C/C++ compiler and CMake installed. For Windows, a simple option is to use [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/) (make sure CLI tools are included) and [CMake](https://cmake.org/download/).

```bash
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_C_COMPILER=cl
```

If you are using Visual Studio, solution files are generated. You can open the solution file with Visual Studio and build the project from there. The `bin` directory would then contain `llama.dll` and `ggml.dll`.
