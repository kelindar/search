#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <include/ggml.h>
#include <include/llama.h>

// Macro to get a procedure address
#ifdef _WIN32
    static HMODULE handle = NULL;
    #define GET_PROC(handle, name) GetProcAddress((HMODULE)(handle), (name))
#else
    static void* handle = NULL;
    #define GET_PROC(handle, name) dlsym(handle, (name))
#endif

// Macro to declare a function pointer
#define DECLARE_LLAMA_FUNC(return_type, func_name, ...) \
    typedef return_type (*func_name##_t)(__VA_ARGS__); \
    static func_name##_t call_##func_name = NULL;

// Macro to load a function pointer
#define LOAD_LLAMA_FUNC(handle, func_name) \
    call_##func_name = (func_name##_t)GET_PROC(handle, #func_name);

// Type definitions for all available LLAMA functions
typedef void* llama_ctx;

// Static function pointers to be loaded
static char error_msg[256];
DECLARE_LLAMA_FUNC(void, llama_free, void* ctx)
DECLARE_LLAMA_FUNC(void, llama_backend_init, void)
DECLARE_LLAMA_FUNC(void, llama_numa_init, enum ggml_numa_strategy)
DECLARE_LLAMA_FUNC(struct llama_model_params, llama_model_default_params, void)
DECLARE_LLAMA_FUNC(struct llama_context_params, llama_context_default_params, void)
DECLARE_LLAMA_FUNC(struct llama_model*, llama_load_model_from_file, const char*, struct llama_model_params)
DECLARE_LLAMA_FUNC(struct llama_context*, llama_new_context_with_model, struct llama_model*, struct llama_context_params)

// Returns the last error message
const char* get_error() {
    return error_msg;
}

// Function to get error messages
const char* get_error_msg(char* error_msg, size_t size) {
#ifdef _WIN32
    DWORD error_code = GetLastError();
    FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        error_code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        error_msg,
        size,
        NULL
    );
#else
    strncpy(error_msg, dlerror(), size - 1);
    error_msg[size - 1] = '\0';
#endif
    return error_msg;
}

// Loads the library and all symbols
int load_library(const char* lib_path) {
#ifdef _WIN32
    HMODULE handle = LoadLibraryA(lib_path);
    if (!handle) {
        snprintf(error_msg, sizeof(error_msg), "Failed to load library: %s", GetLastError());
        return -1;
    }
#else
    void* handle = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        snprintf(error_msg, sizeof(error_msg), "Failed to load library: %s", dlerror());
        return -1;
    }
#endif

    // Load all symbols
    LOAD_LLAMA_FUNC(handle, llama_free)
    LOAD_LLAMA_FUNC(handle, llama_backend_init)
    LOAD_LLAMA_FUNC(handle, llama_numa_init)
    LOAD_LLAMA_FUNC(handle, llama_model_default_params)
    LOAD_LLAMA_FUNC(handle, llama_context_default_params)
    LOAD_LLAMA_FUNC(handle, llama_load_model_from_file)
    LOAD_LLAMA_FUNC(handle, llama_new_context_with_model)

    // Initialize the library
    call_llama_backend_init();
    call_llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);

    printf("Library loaded successfully\n");
    return 0;
}

// loads the model from the given path
llama_ctx load_model(const char* model_path) {
    struct llama_model_params params = call_llama_model_default_params();
    struct llama_model* model = call_llama_load_model_from_file(model_path, params);
    if (!model) {
        snprintf(error_msg, sizeof(error_msg), "Failed to load model: %s", get_error_msg(error_msg, sizeof(error_msg)));
        return NULL;
    }

    // Create a new context with the loaded model
    struct llama_context_params ctx_params = call_llama_context_default_params();
    llama_ctx ctx = call_llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        snprintf(error_msg, sizeof(error_msg), "Failed to create context: %s", get_error_msg(error_msg, sizeof(error_msg)));
        return NULL;
    }

    return ctx;
}