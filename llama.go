package llm

import "fmt"

// Model represents a loaded LLM/Embedding model.
type Model struct {
	handle uintptr
	n_embd int32
}

// New creates a new  model from the given model file.
func New(modelPath string, gpuLayers int) (*Model, error) {
	handle := load_model(modelPath, uint32(gpuLayers))
	if handle == 0 {
		return nil, fmt.Errorf("failed to load model (%s)", modelPath)
	}

	return &Model{
		handle: handle,
		//n_embd: llama_n_embd(handle),
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
	free_model(m.handle)
	m.handle = 0
	return nil
}
