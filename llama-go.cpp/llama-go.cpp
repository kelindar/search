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

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const struct llama_model * model = llama_get_model(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    LOG_INF("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
        // encoder-only model
        if (llama_encode(ctx, batch) < 0) {
            LOG_ERR("%s : failed to encode\n", __func__);
        }
    } else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        // decoder-only model
        if (llama_decode(ctx, batch) < 0) {
            LOG_ERR("%s : failed to decode\n", __func__);
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
}


extern "C" {

    // load the library and initialize the backend
    LLAMA_API void load_library(void){
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
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
    LLAMA_API context_t load_context(model_t model, const uint32_t ctx_size){
        struct llama_context_params params = llama_context_default_params();
        params.n_ctx = ctx_size;
        params.embeddings = true;


        return llama_new_context_with_model(model, params);
        //context_t out = (context_t)malloc(sizeof(context_t));
        //out->model = model;
        //out->ctx = llama_new_context_with_model(model, params);
        //return out;
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

    // embed the text and return the embeddings
    LLAMA_API int embed_text(context_t ctx, const char* text, float* out_embeddings) {
        const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
        model_t model = (model_t)llama_get_model(ctx);
            
            
        // split the prompt into lines
        std::vector<std::string> prompts = split_lines(text, embd_sep);

        // max batch size
        const uint64_t n_batch = llama_n_batch(ctx);

        // tokenize the prompts and trim
        std::vector<std::vector<int32_t>> inputs;
        for (const auto & prompt : prompts) {
            auto inp = ::llama_tokenize(ctx, prompt, true, true);
            if (inp.size() > n_batch) {
                LOG_ERR("%s: number of tokens in input line (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                        __func__, (long long int) inp.size(), (long long int) n_batch);
                return 1;
            }
            inputs.push_back(inp);
        }

        // check if the last token is SEP
        // it should be automatically added by the tokenizer when 'tokenizer.ggml.add_eos_token' is set to 'true'
        for (auto & inp : inputs) {
            if (inp.empty() || inp.back() != llama_token_sep(model)) {
                return 1; // last token is not SEP
            }
        }

        // initialize batch
        const int n_prompts = prompts.size();
        struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

        // count number of embeddings
        int n_embd_count = 0;
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            for (int k = 0; k < n_prompts; k++) {
                n_embd_count += inputs[k].size();
            }
        } else {
            n_embd_count = n_prompts;
        }

        // allocate output
        const int n_embd = llama_n_embd(model);
        std::vector<float> embeddings(n_embd_count * n_embd, 0);
        float * emb = embeddings.data();

        // break into batches
        int e = 0; // number of embeddings already stored
        int s = 0; // number of prompts in current batch
        for (int k = 0; k < n_prompts; k++) {
            // clamp to n_batch tokens
            auto & inp = inputs[k];

            const uint64_t n_toks = inp.size();

            // encode if at capacity
            if (batch.n_tokens + n_toks > n_batch) {
                float * out = emb + e * n_embd;
                batch_decode(ctx, batch, out, s, n_embd, embd_normalize);
                e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
                s = 0;
                llama_batch_clear(batch);
            }

            // add to batch
            batch_add_seq(batch, inp, s);
            s += 1;
        }

        // final batch
        float * out = emb + e * n_embd;
        batch_decode(ctx, batch, out, s, n_embd, embd_normalize);

        
        memcpy(out_embeddings, out, n_embd * sizeof(float));


        // clean up
        llama_batch_free(batch);
        return 0;
    }
}
