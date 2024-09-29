#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
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


// Returns the last error message
static char error_msg[256];
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

// Context structure to hold the loaded model and context
struct context {
    struct llama_context* ctx;
    struct llama_model* model;
    int32_t n_embd;
};
typedef struct context* context_t;

// Static function pointers to be loaded
DECLARE_LLAMA_FUNC(void, llama_backend_init, void)
DECLARE_LLAMA_FUNC(void, llama_numa_init, enum ggml_numa_strategy)
DECLARE_LLAMA_FUNC(struct llama_model_params, llama_model_default_params, void)
DECLARE_LLAMA_FUNC(struct llama_context_params, llama_context_default_params, void)
DECLARE_LLAMA_FUNC(struct llama_model*, llama_load_model_from_file, const char*, struct llama_model_params)
DECLARE_LLAMA_FUNC(struct llama_context*, llama_new_context_with_model, struct llama_model*, struct llama_context_params)
DECLARE_LLAMA_FUNC(void, llama_free_model, struct llama_model*)
DECLARE_LLAMA_FUNC(void, llama_free, struct llama_context*)
DECLARE_LLAMA_FUNC(int32_t, llama_tokenize, const struct llama_model*, const char*, int32_t, llama_token*, int32_t, bool, bool)
DECLARE_LLAMA_FUNC(int32_t, llama_decode, struct llama_context*, struct llama_batch)
DECLARE_LLAMA_FUNC(float*, llama_get_embeddings_seq, struct llama_context*, llama_seq_id)
DECLARE_LLAMA_FUNC(struct llama_batch, llama_batch_init, int32_t, int32_t, int32_t)
DECLARE_LLAMA_FUNC(void, llama_batch_free, struct llama_batch)
DECLARE_LLAMA_FUNC(int32_t, llama_n_embd, struct llama_model*)
DECLARE_LLAMA_FUNC(bool, llama_model_has_encoder, const struct llama_model*)
DECLARE_LLAMA_FUNC(bool, llama_model_has_decoder, const struct llama_model*)

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
    LOAD_LLAMA_FUNC(handle, llama_backend_init)
    LOAD_LLAMA_FUNC(handle, llama_numa_init)
    LOAD_LLAMA_FUNC(handle, llama_model_default_params)
    LOAD_LLAMA_FUNC(handle, llama_context_default_params)
    LOAD_LLAMA_FUNC(handle, llama_load_model_from_file)
    LOAD_LLAMA_FUNC(handle, llama_new_context_with_model)
    LOAD_LLAMA_FUNC(handle, llama_free_model)
    LOAD_LLAMA_FUNC(handle, llama_free)
    LOAD_LLAMA_FUNC(handle, llama_tokenize)
    LOAD_LLAMA_FUNC(handle, llama_decode)
    LOAD_LLAMA_FUNC(handle, llama_get_embeddings_seq)
    LOAD_LLAMA_FUNC(handle, llama_batch_init)
    LOAD_LLAMA_FUNC(handle, llama_batch_free)
    LOAD_LLAMA_FUNC(handle, llama_n_embd)
    LOAD_LLAMA_FUNC(handle, llama_model_has_encoder)
    LOAD_LLAMA_FUNC(handle, llama_model_has_decoder)

    // Initialize the library
    call_llama_backend_init();
    call_llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    return 0;
}

// loads the model from the given path
context_t load_model(const char* model_path, const uint32_t n_ctx) {
    struct llama_model_params params = call_llama_model_default_params();
    struct llama_model* model = call_llama_load_model_from_file(model_path, params);
    if (!model) {
        snprintf(error_msg, sizeof(error_msg), "Failed to load model: %s", get_error_msg(error_msg, sizeof(error_msg)));
        return NULL;
    }

    // Create a new context with the loaded model
    struct llama_context_params ctx_params = call_llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.embeddings = true;
    // ctx_params.flash_attn = true; 

    struct llama_context* ctx = call_llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        snprintf(error_msg, sizeof(error_msg), "Failed to create context: %s", get_error_msg(error_msg, sizeof(error_msg)));
        return NULL;
    }

    // Return a pointer to the context structure
    context_t out = (context_t)malloc(sizeof(struct context));
    out->ctx = ctx;
    out->model = model;
    out->n_embd = call_llama_n_embd(model);
    return out;
}

// deallocates the model
void free_model(context_t context) {
    call_llama_free(context->ctx);
    call_llama_free_model(context->model);
    free(context);
}

// Generates embeddings vector for the given text.
int embed_text(context_t context, const char* text, float* out_embeddings) {
    if (!context || !text || !out_embeddings) {
        snprintf(error_msg, sizeof(error_msg), "Invalid arguments to embed_text");
        return -1;
    }

    struct llama_context* ctx = context->ctx;
    struct llama_model* model = context->model;

    // Step 1: Determine the number of tokens required
    int32_t text_len = strlen(text);
    int32_t n_tokens = call_llama_tokenize(model, text, text_len, NULL, 0, true, true);
    if (n_tokens < 0) {
        n_tokens = -n_tokens; // Tokens needed
    }

    // Step 2: Initialize the batch with the exact number of tokens
    struct llama_batch batch = call_llama_batch_init(n_tokens, 0, 1); // embd=0, n_seq_max=1
    /*if (n_tokens == 0 || batch.n_tokens != n_tokens) {
        snprintf(error_msg, sizeof(error_msg), "unable to init batch for %d tokens, got %d", n_tokens, batch.n_tokens);
        call_llama_batch_free(batch);
        return -1;
    }*/

    // Step 3: Tokenize the text into the batch.token array
    int32_t actual_n_tokens = call_llama_tokenize(model, text, text_len, batch.token, n_tokens, true, true);
    if (actual_n_tokens < 0 || actual_n_tokens != n_tokens) {
        snprintf(error_msg, sizeof(error_msg), "unable to tokenize %d tokens, got %d",  n_tokens, actual_n_tokens);
        call_llama_batch_free(batch);
        return -1;
    }

    /*
    
void llama_batch_add(struct llama_batch& batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool   logits) {
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}

    */

    // Step 4: Assign pos and seq_id
    batch.all_seq_id = 1; // Single sequence
    for (int32_t i = 0; i < n_tokens; i++) {
        batch.pos[i] = i; 
        batch.n_seq_id[i] = 0; // Single sequence
        batch.seq_id[i] = &batch.all_seq_id; // Point to the single sequence ID
        batch.logits[i] = true; // Enable embeddings extraction
        batch.n_tokens++;
    }

    // Step 5: Decode the tokens
    int32_t decode_result = call_llama_decode(ctx, batch);
    if (decode_result < 0) {
        snprintf(error_msg, sizeof(error_msg), "Decoding failed with code %d", decode_result);
        call_llama_batch_free(batch);
        return -1;
    }


    // Step 6: Retrieve the embeddings
    float* embeddings = call_llama_get_embeddings_seq(ctx, 0);
    if (!embeddings) {
        snprintf(error_msg, sizeof(error_msg), "Failed to retrieve embeddings");
        call_llama_batch_free(batch);
        return -1;
    }

    printf("n_embd: %d\n", context->n_embd);

    // Step 8: Copy the embeddings to the output array
    memcpy(out_embeddings, embeddings, context->n_embd * sizeof(float));

    // Clean up
    // TODO: FIX

    // DEBUG: print batch pointers
    printf("batch.token: %p\n", batch.token);
    printf("batch.embd: %p\n", batch.embd);
    printf("batch.pos: %p\n", batch.pos);
    printf("batch.n_seq_id: %p\n", batch.n_seq_id);
    printf("batch.seq_id: %p\n", batch.seq_id);
    printf("batch.logits: %p\n", batch.logits);
    


    /*if (batch.token)    free(batch.token);
    if (batch.embd)     free(batch.embd);
    if (batch.pos)      free(batch.pos);
    if (batch.n_seq_id) free(batch.n_seq_id);
    if (batch.seq_id) {
        for (int i = 0; batch.seq_id[i] != nullptr; ++i) {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)   free(batch.logits);*/


    call_llama_batch_free(batch);
    return 0; 
}