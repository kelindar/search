package llm

import "fmt"

// Model represents a loaded LLM/Embedding model.
type Model struct {
	handle uintptr
	n_embd int32
}

// New creates a new  model from the given model file.
func New(modelPath string, gpuLayers int) (*Model, error) {
	/*cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.load_model(cPath, C.uint32_t(contextSize))
	if handle == nil {
		return nil, fmt.Errorf("failed to load model: %s", C.GoString(C.get_error()))
	}*/

	llama_backend_init()
	llama_numa_init(1) // distribute

	params := llama_model_default_params()
	//println(params)
	/*handle := llama_load_model_from_file(modelPath, &llama_model_params{
		n_gpu_layers: int32(gpuLayers),
		split_mode:   1, // LLAMA_SPLIT_MODE_ROW
		use_mmap:     true,
	})*/

	//params.n_gpu_layers = int32(gpuLayers)
	handle := llama_load_model_from_file(modelPath, params)
	if handle == 0 {
		return nil, fmt.Errorf("failed to load model (%s)", modelPath)
	}

	return &Model{
		handle: handle,
		n_embd: llama_n_embd(handle),
	}, nil
}

func (m *Model) EmbedText(text string) ([]float32, error) {
	/*cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	switch {
	case !bool(m.handle.has_decoder):
		return nil, fmt.Errorf("model does not support decoding")
	case bool(m.handle.has_encoder):
		return nil, fmt.Errorf("encoder/decoder models are not supported")
	}*/

	embeddings := make([]float32, m.n_embd)
	/*if ret := C.embed_text(m.handle, cText, (*C.float)(unsafe.Pointer(&embeddings[0]))); ret != 0 {
		return nil, fmt.Errorf("failed to embed text: %s", C.GoString(C.get_error()))
	}*/

	return embeddings, nil
}

// Close closes the model and releases any resources associated with it.
func (m *Model) Close() error {
	llama_free_model(m.handle)
	m.handle = 0
	return nil
}
