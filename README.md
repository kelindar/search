<p align="center">
<img width="300" height="100" src=".github/logo.png" border="0" alt="kelindar/search">
<br>
<img src="https://img.shields.io/github/go-mod/go-version/kelindar/search" alt="Go Version">
<a href="https://pkg.go.dev/github.com/kelindar/search"><img src="https://pkg.go.dev/badge/github.com/kelindar/search" alt="PkgGoDev"></a>
<a href="https://goreportcard.com/report/github.com/kelindar/search"><img src="https://goreportcard.com/badge/github.com/kelindar/search" alt="Go Report Card"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
<a href="https://coveralls.io/github/kelindar/search"><img src="https://coveralls.io/repos/github/kelindar/search/badge.svg" alt="Coverage"></a>
</p>

# Vector search and embeddings

An exact cosine similarity index for small and medium datasets, with optional local and cloud embedding providers. Requires Go 1.27 or newer.

| Package | Purpose |
|---|---|
| `github.com/kelindar/search` | Vector indexing, search, and persistence. No native library required. |
| `github.com/kelindar/search/llama` | Local GGUF embeddings through llama.cpp and purego. |
| `github.com/kelindar/search/openai` | Text embeddings through OpenAI-compatible APIs, including OpenRouter. |

Each library has its own `go.mod` and `go.sum`. Core `search` has no dependency on either embedding module or `purego`. The `openai` module uses only the standard library at runtime; `llama` owns the native inference dependency. Testify and its YAML dependency are used only by tests. Import paths and APIs are unchanged.

The interactive `example` has a separate development module with local replacements for `search` and `llama`. Native quality evaluation lives under `llama/internal/eval`, so neither introduces provider dependencies into core search.

## Search with your own vectors

```go
index := search.NewIndex[string]()
index.Add([]float32{1, 0}, "cat")
index.Add([]float32{0, 1}, "dog")
results := index.Search([]float32{1, 0}, 1)
fmt.Println(results[0].Value) // cat
```

`Add` normalizes the vector in place and retains its backing array. Do not modify it afterward. `Search` normalizes the query in place. Results contain cosine similarity in descending order, so larger `Relevance` values are better. Use nonzero, finite vectors from the same model with the same dimensions for all documents and queries. Changing models requires re-embedding the dataset, even if dimensions match.

Existing `ReadFile`, `WriteFile`, `ReadFrom`, and `WriteTo` formats remain unchanged. Search scans every vector; large datasets may need a different indexing algorithm. Concurrent searches require separate query slices and no concurrent index writes.

## Local embeddings

Import `github.com/kelindar/search/llama` and install the native library as described below.

```go
model, err := llama.New("dist/MiniLM-L6-v2.Q8_0.gguf", 0)
if err != nil {
    return err
}
defer model.Close()
vector, err := model.EmbedText(ctx, "Your text here")
```

Zero GPU layers uses the CPU. The library initializes on the first valid constructor call, never on import; initialization failures are returned as errors and cached for the process lifetime. Configure library paths before constructing a model.

Embedding calls on a model may run concurrently. Finish all calls and close any explicitly created `model.Context(size)` values before closing the model. Individual contexts must not be shared concurrently. Contexts retain `Tokens()` accounting. Cancellation is checked around native inference and between batch items; it cannot interrupt native inference already in progress.

## Cloud embeddings

Import `github.com/kelindar/search/openai`. Supply credentials and a model explicitly.

```go
client, err := openai.New(os.Getenv("OPENAI_API_KEY"), "text-embedding-3-small")
if err != nil {
    return err
}
vector, err := client.EmbedText(ctx, "Your text here")
vectors, err := client.EmbedBatch(ctx, []string{"first document", "second document"})
```

For OpenRouter, use its base URL and model identifier:

```go
client, err := openai.New(os.Getenv("OPENROUTER_API_KEY"), "openai/text-embedding-3-small",
    openai.BaseURL("https://openrouter.ai/api/v1"),
)
```

Optional configuration uses `Dimensions(n)`, `Headers(http.Header)`, and `HTTPClient(*http.Client)`. Defaults are the OpenAI `/v1` endpoint, provider-default dimensions, no extra headers, and a 60-second HTTP timeout. A supplied HTTP client controls its own timeout. Headers are copied and cannot override authorization or content type. Clients may be shared concurrently; do not mutate a supplied HTTP client during use.

Batches use one HTTP request and return vectors in input order. Returned vectors belong to the caller. Empty batches do no work; failures return no partial results. There are no automatic retries or batch splitting. Local `EmbedBatch` uses sequential inference with the same result contract. Cloud support covers text, not multimodal inputs or provider-specific routing.

## Custom providers

Any function returning `[]float32` can supply vectors to the index, without importing either provider package. Applications that swap providers can define their own small interface:

```go
type Embedder interface {
    EmbedText(context.Context, string) ([]float32, error)
}
```

Both built-in providers satisfy this interface. See [the custom-provider example](index_codec_test.go).

## Migration and tests

Replace `search.NewVectorizer(path, layers)` with `llama.New(path, layers)`, `search.Vectorizer` with `llama.Vectorizer`, and `search.Context` with `llama.Context`. Pass a `context.Context` as the first argument to `EmbedText`, including calls on explicit local contexts. Index APIs and saved files require no migration.

Run `go test ./...` separately from the repository root, `llama`, `openai`, and `example`. Root `./...` does not cross module boundaries. Native model and quality tests are opt-in via `go test -tags integration ./...` from `llama`; they require the native library on its platform search path and the repository's MiniLM model and evaluation fixtures. The native source is pinned to llama.cpp v0.4.1. The binaries in [dist](dist) predate that pin; build the wrapper from source for the updated version. Build instructions below apply only to local embeddings.

Nested modules are released with their own tags: `llama/vX.Y.Z` and `openai/vX.Y.Z`. Root releases continue to use `vX.Y.Z`. Tests and module maintenance run with `GOWORK=off` in CI to verify isolation.

## 🛠 Compile library

First, clone the repository and its submodules with the following commands. This checks out the pinned llama.cpp source and retrieves the model and vector fixtures.

```bash
git submodule update --init --recursive
git lfs pull
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
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc ..
cmake --build . --config Release --target llama_go --parallel
```

This should generate `libllama_go.so` that statically links everything necessary. You can also install the library by coping it into `/usr/lib`.

### Compile on macOS

Make sure you have Xcode Command Line Tools and CMake installed:

```bash
xcode-select --install
brew install cmake
```

Then compile the library:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --target llama_go --parallel
```

This generates `libllama_go.dylib` in `build/lib`. If you run the Go tests or example from the repository root, point the dynamic loader to that directory:

```bash
cd llama
DYLD_LIBRARY_PATH="$(pwd)/../build/lib" go test -tags integration ./...
```

### Compile on Windows

Make sure you have a C/C++ compiler and CMake installed. For Windows, a simple option is to use [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/) (make sure CLI tools are included) and [CMake](https://cmake.org/download/).

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --target llama_go --parallel
```

If you are using Visual Studio, solution files are generated. You can open the solution file with Visual Studio and build the project from there. The `bin/Release` directory contains `llama_go.dll`. Add that directory to `PATH` before running local embeddings or integration tests.

### GPU and other options

To enable GPU support (e.g. Vulkan), you'll need to add an appropriate flag to the CMake command, please refer to refer to the [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan) build documentation for more details. For example, to compile with Vulkan support on Windows make sure Vulkan SDK is installed and then run the following commands:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON ..
cmake --build . --config Release --target llama_go --parallel
```
