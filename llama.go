package llm

import "fmt"

// Model represents a loaded LLM/Embedding model.
type Model struct {
	handle uintptr
	n_embd int32
}

type Context struct {
	parent *Model
	handle uintptr
}

// New creates a new  model from the given model file.
func New(modelPath string, gpuLayers int) (*Model, error) {
	handle := load_model(modelPath, uint32(gpuLayers))
	if handle == 0 {
		return nil, fmt.Errorf("failed to load model (%s)", modelPath)
	}

	return &Model{
		handle: handle,
		n_embd: embed_size(handle),
	}, nil
}

// Close closes the model and releases any resources associated with it.
func (m *Model) Close() error {
	free_model(m.handle)
	m.handle = 0
	return nil
}

// Context creates a new context of the given size.
func (m *Model) Context(size int) *Context {
	return &Context{
		parent: m,
		handle: load_context(m.handle, uint32(size)),
	}
}

// EmbedText embeds the given text using the model.
func (m *Model) EmbedText(text string) ([]float32, error) {
	ctx := m.Context(0)
	defer ctx.Close()
	return ctx.EmbedText(text)
}

// Close closes the context and releases any resources associated with it.
func (ctx *Context) Close() error {
	free_context(ctx.handle)
	ctx.handle = 0
	return nil
}

// --------------------------------- Context ---------------------------------

// EmbedText embeds the given text using the model.
func (ctx *Context) EmbedText(text string) ([]float32, error) {
	switch {
	case ctx.handle == 0 || ctx.parent.handle == 0:
		return nil, fmt.Errorf("context is not initialized")
	case ctx.parent.n_embd <= 0:
		return nil, fmt.Errorf("model does not support embedding")
	}

	embeddings := make([]float32, ctx.parent.n_embd)

	ret := embed_text(ctx.handle, text, embeddings)
	switch ret {
	case 0:
		return embeddings, nil
	case 1:
		return nil, fmt.Errorf("last token in the prompt is not SEP")
	default:
		return nil, fmt.Errorf("failed to embed text (code=%d)", ret)
	}

	/*if ret := C.embed_text(m.handle, cText, (*C.float)(unsafe.Pointer(&embeddings[0]))); ret != 0 {
		return nil, fmt.Errorf("failed to embed text: %s", C.GoString(C.get_error()))
	}*/

}
