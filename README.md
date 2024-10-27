# Semantic Search

This library was created to provide an **easy and efficient solution for embedding and vector search**, making it perfect for small to medium-scale projects that still need some **serious semantic power**. It’s built around a simple idea: if your dataset is small enough, you can achieve accurate results with brute-force techniques, and with some smart optimizations like **SIMD**, you can keep things fast and lean.

The library’s strength lies in its simplicity and support for **GGUF BERT models**, letting you leverage sophisticated embeddings without getting bogged down by the complexities of traditional search systems. It offers **GPU acceleration**, enabling quick computations on supported hardware. If your dataset has fewer than 100,000 entries, this library is a great fit for integrating semantic search into your Go applications with minimal hassle.

![demo](./.github/demo.gif)

## 🚀 Key Features

- **llama.cpp without cgo**: The library is built to work with [llama.cpp](https://github.com/ggerganov/llama.cpp) without using cgo. Instead, it relies on [purego](https://github.com/ebitengine/purego) , which allows calling shared C libraries directly from Go code without the need for cgo. This design significantly simplifies the integration, deployment, and cross-compilation, making it easier to build Go applications that interface with native libraries.
- **Support for BERT Models**: The library supports BERT models via [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5423). Vast variations of BERT models can be used, as long as they are using GGUF format.
- **Precompiled Binaries with Vulkan GPU Support**: Available for Windows and Linux in the [dist](dist) directory, compiled with Vulkan for GPU acceleration. However, you can compile the library yourself with or without GPU support.
- **Search Index for Embeddings**: The library supports the creation of a search index from computed embeddings, which can be saved to disk and loaded later. This feature is suitable for basic vector-based searches in small-scale applications, but it may face efficiency challenges with large datasets due to the use of brute-force techniques.

## Limitations

While simple vector search excels in small-scale applications,avoid using this library if you have the following requirements.

- **Large Datasets**: The current implementation is designed for small-scale applications, and datasets exceeding 100,000 entries may suffer from performance bottlenecks due to the brute-force search approach. For larger datasets, approximate nearest neighbor (ANN) algorithms and specialized data structures should be considered for efficiency.
- **Complex Query Requirements**: The library focuses on simple vector similarity search and does not support advanced query capabilities like multi-field filtering, fuzzy matching, or SQL-like operations that are common in more sophisticated search engines.
- **High-Dimensional Complex Embeddings**: Large language models (LLMs) generate embeddings that are both high-dimensional and computationally intensive. Handling these embeddings in real-time can be taxing on the system unless sufficient GPU resources are available and optimized for low-latency inference.

## 📚 How to Use the Library

This example demonstrates how to use the library to generate embeddings for text and perform a simple vector search. The code snippet below shows how to load a model, generate embeddings for text, create a search index, and perform a search.

1. **Install library**: Precompiled binaries for Windows and Linux are provided in the [dist](dist) directory. If your target architecture or platform isn't covered by these binaries, you'll need to compile the library from the source. Drop these binaries in `/usr/lib` or equivalent.

1. **Load a model**: The `search.NewVectorizer` function initializes a model using a GGUF file. This example loads the _MiniLM-L6-v2.Q8_0.gguf_ model. The second parameter, indicates the number of GPU layers to enable (0 for CPU only).

```go
m, err := search.NewVectorizer("../dist/MiniLM-L6-v2.Q8_0.gguf", 0)
if err != nil {
    // handle error
}
defer m.Close()
```

3. **Generate text embeddings**: The `EmbedText` method is used to generate vector embeddings for a given text input. This converts your text into a dense numerical vector representation given the model you loaded in the previous step.

```go
embedding, err := m.EmbedText("Your text here")
```

4. **Create an index and adding vectors**: Create a new index using `search.NewIndex`. The type parameter `[string]` in this example specifies that each vector is associated with a string value. You can add multiple vectors with corresponding labels.

```go
index := search.NewIndex[string]()
index.Add(embedding, "Your text here")
```

5. **Search the index**: Perform a search using the `Search` method, which takes an embedding vector and a number of results to retrieve. This example searches for the 10 most relevant results and prints them along with their relevance scores.

```go
results := index.Search(embedding, 10)
for _, r := range results {
    fmt.Printf("Result: %s (Relevance: %.2f)\n", r.Value, r.Relevance)
}
```

## 🛠 Compile library

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
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc ..
cmake --build . --config Release
```

This should generate `libllama.so` and `libggml.so` that you can use. You can also install the library by coping it into `/usr/lib`.

### Compile on Windows

Make sure you have a C/C++ compiler and CMake installed. For Windows, a simple option is to use [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/) (make sure CLI tools are included) and [CMake](https://cmake.org/download/).

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

If you are using Visual Studio, solution files are generated. You can open the solution file with Visual Studio and build the project from there. The `bin` directory would then contain `llama.dll` and `ggml.dll`.

### GPU and other options

To enable GPU support (e.g. Vulkan), you'll need to add an appropriate flag to the CMake command, please refer to refer to the [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#vulkan) build documentation for more details. For example, to compile with Vulkan support on Windows make sure Vulkan SDK is installed and then run the following commands:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON ..
cmake --build . --config Release
```
