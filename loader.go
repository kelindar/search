package llm

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ebitengine/purego"
)

// libptr is a pointer to the loaded dynamic library.
var libptr uintptr

/*
void llama_backend_init(void);
void llama_numa_init(enum ggml_numa_strategy numa);
struct llama_model_params llama_model_default_params(void);
struct llama_context_params llama_context_default_params(void);
struct llama_model * llama_load_model_from_file(const char * path_model, struct llama_model_params params);
struct llama_context * llama_new_context_with_model(struct llama_model * model, struct llama_context_params params);
void llama_free_model(struct llama_model * model);
void llama_free(struct llama_context * ctx);
struct llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
void llama_batch_free(struct llama_batch batch);
int32_t llama_tokenize(const struct llama_model * model, const char * text, int32_t text_len, llama_token * tokens, int32_t n_tokens_max, bool add_special, bool parse_special);
int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch);
int32_t llama_encode(struct llama_context * ctx, struct llama_batch batch);
void llama_kv_cache_clear(struct llama_context * ctx);
float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);

int32_t llama_n_embd(const struct llama_model * model);
bool llama_model_has_encoder(const struct llama_model * model);
bool llama_model_has_decoder(const struct llama_model * model);
enum llama_pooling_type llama_pooling_type(const struct llama_context * ctx);
*/
var llama_backend_init func()
var llama_numa_init func(numa int)
var llama_model_default_params func() *llama_model_params
var llama_context_default_params func() llama_context_params
var llama_load_model_from_file func(path_model string, params *llama_model_params) uintptr
var llama_new_context_with_model func(model uintptr, params llama_context_params) uintptr
var llama_free_model func(model uintptr)
var llama_free func(ctx uintptr)
var llama_batch_init func(n_tokens, embd, n_seq_max int32) llama_batch
var llama_batch_free func(batch llama_batch)
var llama_tokenize func(model uintptr, text string, text_len int32, tokens []int32, n_tokens_max int32, add_special, parse_special bool) int32
var llama_decode func(ctx uintptr, batch llama_batch) int32
var llama_encode func(ctx uintptr, batch llama_batch) int32
var llama_kv_cache_clear func(ctx uintptr)
var llama_get_embeddings_ith func(ctx uintptr, i int32) []float32
var llama_get_embeddings_seq func(ctx uintptr, seq_id int32) []float32
var llama_n_embd func(model uintptr) int32
var llama_model_has_encoder func(model uintptr) bool
var llama_model_has_decoder func(model uintptr) bool
var llama_pooling_type func(ctx uintptr) int32

func init() {
	libpath, err := findLlama()
	if err != nil {
		panic(err)
	}
	if libptr, err = load(libpath); err != nil {
		panic(err)
	}

	// Load the library functions
	purego.RegisterLibFunc(&llama_backend_init, libptr, "llama_backend_init")
	purego.RegisterLibFunc(&llama_numa_init, libptr, "llama_numa_init")
	purego.RegisterLibFunc(&llama_model_default_params, libptr, "llama_model_default_params")
	purego.RegisterLibFunc(&llama_load_model_from_file, libptr, "llama_load_model_from_file")
	/*purego.RegisterLibFunc(&llama_context_default_params, libptr, "llama_context_default_params")
	purego.RegisterLibFunc(&llama_new_context_with_model, libptr, "llama_new_context_with_model")
	purego.RegisterLibFunc(&llama_free_model, libptr, "llama_free_model")
	purego.RegisterLibFunc(&llama_free, libptr, "llama_free")
	purego.RegisterLibFunc(&llama_batch_init, libptr, "llama_batch_init")
	purego.RegisterLibFunc(&llama_batch_free, libptr, "llama_batch_free")
	purego.RegisterLibFunc(&llama_tokenize, libptr, "llama_tokenize")
	purego.RegisterLibFunc(&llama_decode, libptr, "llama_decode")
	purego.RegisterLibFunc(&llama_encode, libptr, "llama_encode")
	purego.RegisterLibFunc(&llama_kv_cache_clear, libptr, "llama_kv_cache_clear")
	purego.RegisterLibFunc(&llama_get_embeddings_ith, libptr, "llama_get_embeddings_ith")
	purego.RegisterLibFunc(&llama_get_embeddings_seq, libptr, "llama_get_embeddings_seq")
	purego.RegisterLibFunc(&llama_n_embd, libptr, "llama_n_embd")
	purego.RegisterLibFunc(&llama_model_has_encoder, libptr, "llama_model_has_encoder")
	purego.RegisterLibFunc(&llama_model_has_decoder, libptr, "llama_model_has_decoder")
	purego.RegisterLibFunc(&llama_pooling_type, libptr, "llama_pooling_type")*/

}

// --------------------------------- Structs ---------------------------------

type llama_model_params struct {
	n_gpu_layers                int32     // number of layers to store in VRAM
	split_mode                  int       // how to split the model across multiple GPUs
	main_gpu                    int32     // the GPU that is used for the entire model
	tensor_split                []float32 // proportion of the model (layers or rows) to offload to each GPU
	rpc_servers                 string    // comma separated list of RPC servers to use for offloading
	progress_callback           uintptr   // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
	progress_callback_user_data uintptr   // context pointer passed to the progress callback
	kv_overrides                uintptr   // override key-value pairs of the model meta data
	vocab_only                  bool      // only load the vocabulary, no weights
	use_mmap                    bool      // use mmap if possible
	use_mlock                   bool      // force system to keep model in RAM
	check_tensors               bool      // validate model tensor data

	/*
		int32_t n_gpu_layers; // number of layers to store in VRAM
		enum llama_split_mode split_mode; // how to split the model across multiple GPUs

		// main_gpu interpretation depends on split_mode:
		// LLAMA_SPLIT_MODE_NONE: the GPU that is used for the entire model
		// LLAMA_SPLIT_MODE_ROW: the GPU that is used for small tensors and intermediate results
		// LLAMA_SPLIT_MODE_LAYER: ignored
		int32_t main_gpu;

		// proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
		const float * tensor_split;

		// comma separated list of RPC servers to use for offloading
		const char * rpc_servers;

		// Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
		// If the provided progress_callback returns true, model loading continues.
		// If it returns false, model loading is immediately aborted.
		llama_progress_callback progress_callback;

		// context pointer passed to the progress callback
		void * progress_callback_user_data;

		// override key-value pairs of the model meta data
		const struct llama_model_kv_override * kv_overrides;

		// Keep the booleans together to avoid misalignment during copy-by-value.
		bool vocab_only;    // only load the vocabulary, no weights
		bool use_mmap;      // use mmap if possible
		bool use_mlock;     // force system to keep model in RAM
		bool check_tensors; // validate model tensor data
	*/
}

type llama_context_params struct {
	/*
		uint32_t n_ctx;             // text context, 0 = from model
		uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
		uint32_t n_ubatch;          // physical maximum batch size
		uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
		int32_t  n_threads;         // number of threads to use for generation
		int32_t  n_threads_batch;   // number of threads to use for batch processing

		enum llama_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
		enum llama_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
		enum llama_attention_type    attention_type;    // attention type to use for embeddings

		// ref: https://github.com/ggerganov/llama.cpp/pull/2054
		float    rope_freq_base;   // RoPE base frequency, 0 = from model
		float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
		float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
		float    yarn_attn_factor; // YaRN magnitude scaling factor
		float    yarn_beta_fast;   // YaRN low correction dim
		float    yarn_beta_slow;   // YaRN high correction dim
		uint32_t yarn_orig_ctx;    // YaRN original context size
		float    defrag_thold;     // defragment the KV cache if holes/size > thold, < 0 disabled (default)

		ggml_backend_sched_eval_callback cb_eval;
		void * cb_eval_user_data;

		enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
		enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

		// Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
		// TODO: move at the end of the struct
		bool logits_all;  // the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
		bool embeddings;  // if true, extract embeddings (together with logits)
		bool offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU
		bool flash_attn;  // whether to use flash attention [EXPERIMENTAL]
		bool no_perf;     // whether to measure performance timings

		// Abort callback
		// if it returns true, execution of llama_decode() will be aborted
		// currently works only with CPU execution
		ggml_abort_callback abort_callback;
		void *              abort_callback_data;
	*/
}

// Input data for llama_decode
// A llama_batch object can contain input about one or many sequences
// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
type llama_batch struct {
	n_tokens int32
	token    []int32   // the token ids of the input (used when embd is NULL)
	embd     []float32 // the embeddings of the input (used when token is NULL)
	pos      []int32   // the positions of the respective token in the sequence
	n_seq_id []int32   // the number of tokens in each sequence
	seq_id   [][]int32 // the sequence to which the respective token belongs
	logits   []int8    // the logits (and/or the embeddings) for the respective token

	/*
	   typedef struct llama_batch {
	       int32_t n_tokens;

	       llama_token  *  token;
	       float        *  embd;
	       llama_pos    *  pos;
	       int32_t      *  n_seq_id;
	       llama_seq_id ** seq_id;
	       int8_t       *  logits; // TODO: rename this to "output"

	       // NOTE: helpers for smooth API transition - can be deprecated in the future
	       //       for future-proof code, use the above fields instead and ignore everything below
	       //
	       // pos[i] = all_pos_0 + i*all_pos_1
	       //
	       llama_pos    all_pos_0;  // used if pos == NULL
	       llama_pos    all_pos_1;  // used if pos == NULL
	       llama_seq_id all_seq_id; // used if seq_id == NULL
	   } llama_batch;
	*/
}

// --------------------------------- Library Lookup ---------------------------------

// findLlama searches for the dynamic library in standard system paths.
func findLlama() (string, error) {
	switch runtime.GOOS {
	case "windows":
		return findLibrary("llama.dll", runtime.GOOS)
	case "darwin":
		return findLibrary("libllama.dylib", runtime.GOOS)
	default:
		return findLibrary("libllama.so", runtime.GOOS)
	}
}

// findLibrary searches for a dynamic library by name across standard system paths.
// It returns the full path to the library if found, or an error listing all searched paths.
func findLibrary(libName, goos string, dirs ...string) (string, error) {
	libExt, commonPaths := findLibDirs(goos)
	dirs = append(dirs, commonPaths...)

	// Append the correct extension if missing
	if !strings.HasSuffix(libName, libExt) {
		libName += libExt
	}

	// Include current working directory
	if cwd, err := os.Getwd(); err == nil {
		dirs = append(dirs, cwd)
	}

	// Iterate through directories and search for the library
	searched := make([]string, 0, len(dirs))
	for _, dir := range dirs {
		filename := filepath.Join(dir, libName)
		searched = append(searched, filename)
		if fi, err := os.Stat(filename); err == nil && !fi.IsDir() {
			return filename, nil // Library found
		}
	}

	// Construct error message listing all searched paths
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Library '%s' not found, checked following paths:\n", libName))
	for _, path := range searched {
		sb.WriteString(fmt.Sprintf(" - %s\n", path))
	}

	return "", errors.New(sb.String())
}

// findLibDirs returns the library extension, relevant environment path, and common library directories based on the OS.
func findLibDirs(goos string) (string, []string) {
	switch goos {
	case "windows":
		systemRoot := os.Getenv("SystemRoot")
		return ".dll", append(
			filepath.SplitList(os.Getenv("PATH")),
			filepath.Join(systemRoot, "System32"),
			filepath.Join(systemRoot, "SysWOW64"),
		)
	case "darwin":
		return ".dylib", append(
			filepath.SplitList(os.Getenv("DYLD_LIBRARY_PATH")),
			"/usr/lib",
			"/usr/local/lib",
		)
	default: // Unix/Linux
		return ".so", append(
			filepath.SplitList(os.Getenv("LD_LIBRARY_PATH")),
			"/lib",
			"/usr/lib",
			"/usr/local/lib",
		)
	}
}
