#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include <vector>

typedef struct llama_model* model_t;
typedef struct llama_context* context_t;

std::string embd_sep = "\n";
int32_t embd_normalize = 2; // normalisation for embendings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)

static std::vector<std::string> split_lines(const std::string & s, const std::string & separator = "\n") {
    std::vector<std::string> lines;
    size_t start = 0;
    size_t end = s.find(separator);

    while (end != std::string::npos) {
        lines.push_back(s.substr(start, end - start));
        start = end + separator.length();
        end = s.find(separator, start);
    }

    lines.push_back(s.substr(start)); // Add the last part

    return lines;
}

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

static int batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const struct llama_model * model = llama_get_model(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
        if (llama_encode(ctx, batch) < 0) { // encoder-only model
            return -1;
        }
    } else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        if (llama_decode(ctx, batch) < 0) { // decoder-only model
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
            // try to get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            GGML_ASSERT(embd != NULL && "failed to get token embeddings");
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
        }

        float * out = output + embd_pos * n_embd;
        llama_embd_normalize(embd, out, n_embd, embd_norm);
    }
    return 0;
}

extern "C" {

    // load the library and initialize the backend
    LLAMA_API void load_library(ggml_log_level desired){
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);

        // Set the log level
        auto desired_ptr = new ggml_log_level;
        *desired_ptr = desired;
        llama_log_set([](ggml_log_level level, const char* text, void* user_data) {
            if (level < *(ggml_log_level*)user_data) {
                return; // noop
            }
            
            fputs(text, stderr);
            fflush(stderr);
        }, desired_ptr);
    }

    // load the model from the file
    LLAMA_API model_t load_model(const char * path_model, const uint32_t n_gpu_layers){
        struct llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = n_gpu_layers;

        return llama_load_model_from_file(path_model, params);
    }

    // free the model and all the resources
    LLAMA_API void free_model(model_t model){
        llama_free_model(model);
    }

    // create a context with the model and the context size
    LLAMA_API context_t load_context(model_t model, const uint32_t ctx_size, const bool embeddings){
        struct llama_context_params params = llama_context_default_params();
        params.n_ctx = ctx_size;
        params.embeddings = embeddings;
        return llama_new_context_with_model(model, params);
    }

    // free the context and all the resources
    LLAMA_API void free_context(context_t ctx){
        llama_free(ctx);
    }

    // get the embeddings size, if the model doesn't support embeddings, return -1
    LLAMA_API int32_t embed_size(model_t model){
        if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
            return -1; // embeddings are not supported
        }
        return llama_n_embd(model);
    }

    // embed the text and return the embeddings.
    LLAMA_API int embed_text(context_t ctx, const char* text, float* out_embeddings, uint32_t* out_tokens){
        const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
        model_t model = (model_t)llama_get_model(ctx);
        const uint64_t n_batch = llama_n_batch(ctx);

        // Tokenize the prompt
        auto inp = ::llama_tokenize(ctx, text, true, true);
        *out_tokens = inp.size();
        if (inp.size() > n_batch) {
            printf("Number of tokens exceeds batch size, increase batch size\n");
            return 1; // Number of tokens exceeds batch size, increase batch size
        }

        // Check if the last token is SEP
        if (inp.empty() || inp.back() != llama_token_sep(model)) {
            return 2; // Last token is not SEP
        }

        // Initialize batch
        struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
        batch_add_seq(batch, inp, 0);

        // Decode batch and write embeddings directly to out_embeddings
        const int n_embd = llama_n_embd(model);
        if (batch_decode(ctx, batch, out_embeddings, 1, n_embd, embd_normalize) != 0) {
            llama_batch_free(batch);
            return 3; // Decoding failed
        }

        // Clean up
        llama_batch_free(batch);
        return 0;
    }
}
