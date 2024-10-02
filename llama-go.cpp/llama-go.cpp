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

    // Generate text predictions based on a given prompt
    LLAMA_API int complete_text(context_t ctx, const char* prompt, char* output_text, uint32_t max_output_length, uint32_t n_predict) {
        model_t model = (model_t)llama_get_model(ctx);
        const int n_ctx = llama_n_ctx(ctx);
        const uint64_t n_batch = llama_n_batch(ctx);

        // Tokenize the prompt
        std::vector<llama_token> tokens_list;
        tokens_list = ::llama_tokenize(ctx, prompt, true);

        // Check if context size is sufficient
        if (tokens_list.size() + n_predict > n_ctx) {
            return 1; // Error: context size exceeded
        }

        // Initialize batch
        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        for (size_t i = 0; i < tokens_list.size(); i++) {
            llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        // Evaluate the initial prompt
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return 2; // Decoding failed
        }

        // Set up the sampler (using greedy sampling)
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        llama_sampler* smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50)); // Top 50 tokens
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1)); // Top 90% probability
        llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8)); // Temperature 0.8

        // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
        // this sampler will be responsible to select the actual token
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        // Main loop to generate tokens
        int n_cur = tokens_list.size();
        int n_decode = 0;
        std::vector<llama_token> output_tokens;

        while (n_decode < n_predict) {
            // Sample the next token
            const llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // Check for end of generation
            if (llama_token_is_eog(model, new_token_id)) {
                break;
            }

            // Add token to output
            output_tokens.push_back(new_token_id);

            // Prepare the next batch
            llama_batch_clear(batch);

            // Push new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

            n_decode += 1;
            n_cur += 1;

            // Evaluate the current batch
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                llama_sampler_free(smpl);
                return 3; // Decoding failed
            }

            // Check if context size exceeded
            if (n_cur >= n_ctx) {
                break;
            }
        }

        // Convert output tokens to text
        std::string generated_text;
        for (auto id : output_tokens) {
            generated_text += llama_token_to_piece(ctx, id);
        }

        // Copy generated text to output buffer
        if (generated_text.size() >= max_output_length) {
            llama_batch_free(batch);
            llama_sampler_free(smpl);
            return 4; // Output buffer too small
        }
        strcpy(output_text, generated_text.c_str());

        // Clean up
        llama_batch_free(batch);
        llama_sampler_free(smpl);
        return 0;
    }

}
