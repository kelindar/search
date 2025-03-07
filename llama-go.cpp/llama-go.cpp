#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include <vector>

typedef struct llama_model* model_t;
typedef struct llama_context* context_t;

std::string embd_sep = "\n";
int32_t embd_normalize = 2; // normalization (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

static int batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const struct llama_model * model = llama_get_model(ctx);

    // Clear previous KV cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // Run model
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
        if (llama_encode(ctx, batch) < 0) { // Encoder-only model
            return -1;
        }
    } else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        if (llama_decode(ctx, batch) < 0) { // Decoder-only model
            return -1;
        }
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = nullptr;
        int embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // Get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            GGML_ASSERT(embd != NULL && "Failed to get token embeddings");
        } else {
            // Get sequence embeddings (for pooled cases)
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            GGML_ASSERT(embd != NULL && "Failed to get sequence embeddings");
        }

        float * out = output + embd_pos * n_embd;
        common_embd_normalize(embd, out, n_embd, embd_norm);
    }
    return 0;
}

extern "C" {

    // Load the library and initialize the backend
    LLAMA_API void load_library(ggml_log_level desired) {
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);

        // Set the log level
        auto desired_ptr = new ggml_log_level;
        *desired_ptr = desired;
        llama_log_set([](ggml_log_level level, const char* text, void* user_data) {
            if (level < *(ggml_log_level*)user_data) {
                return; // No-op
            }
            
            fputs(text, stderr);
            fflush(stderr);
        }, desired_ptr);
    }

    // Load the model from the file
    LLAMA_API model_t load_model(const char * path_model, const uint32_t n_gpu_layers) {
        struct llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = n_gpu_layers;

        return llama_model_load_from_file(path_model, params); // Updated function
    }

    // Free the model and all resources
    LLAMA_API void free_model(model_t model) {
        llama_model_free(model); // Updated function
    }

    // Create a context with the model and the specified context size
    LLAMA_API context_t load_context(model_t model, const uint32_t ctx_size, const bool embeddings) {
        struct llama_context_params params = llama_context_default_params();
        params.n_ctx = ctx_size;
        params.embeddings = embeddings; // Corrected field name

        return llama_init_from_model(model, params); // Updated function
    }

    // Free the context and all resources
    LLAMA_API void free_context(context_t ctx) {
        llama_free(ctx);
    }

    // Get the embedding size, return -1 if model doesn't support embeddings
    LLAMA_API int32_t embed_size(model_t model) {
        if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
            return -1; // Embeddings not supported for encoder-decoder models
        }
        return llama_model_n_embd(model);
    }

    // Embed the text and return the embeddings
    LLAMA_API int embed_text(context_t ctx, const char* text, float* out_embeddings, uint32_t* out_tokens) {
        const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
        model_t model = (model_t)llama_get_model(ctx);
        const uint64_t n_batch = llama_n_batch(ctx);

        // Tokenize the input text
        auto inp = common_tokenize(ctx, text, true, true);
        *out_tokens = inp.size();
        if (inp.size() > n_batch) {
            printf("Number of tokens exceeds batch size, increase batch size\n");
            return 1; // Number of tokens exceeds batch size
        }

        // Check if the last token is SEP
        if (inp.empty() || inp.back() != llama_vocab_sep(llama_model_get_vocab(model))) {
            return 2; // Last token is not SEP
        }

        // Initialize batch
        struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
        batch_add_seq(batch, inp, 0);

        // Decode batch and store embeddings in out_embeddings
        const int n_embd = llama_model_n_embd(model);
        if (batch_decode(ctx, batch, out_embeddings, 1, n_embd, embd_normalize) != 0) {
            llama_batch_free(batch);
            return 3; // Decoding failed
        }

        // Clean up
        llama_batch_free(batch);
        return 0;
    }
}
