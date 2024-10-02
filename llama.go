package llm

import (
	"fmt"
	"sync/atomic"
)

// Model represents a loaded LLM/Embedding model.
type Model struct {
	handle uintptr
	n_embd int32
}

type Context struct {
	parent *Model
	handle uintptr
	tokens atomic.Uint64
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
		handle: load_context(m.handle, uint32(size), false),
	}
}

// EmbedText embeds the given text using the model.
func (m *Model) EmbedText(text string) ([]float32, error) {
	//ctx := m.Context(0)
	//defer ctx.Close()

	ctx := &Context{
		parent: m,
		handle: load_context(m.handle, 0, true),
	}
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

// Tokens returns the number of tokens processed by the context.
func (ctx *Context) Tokens() uint {
	return uint(ctx.tokens.Load())
}

// EmbedText embeds the given text using the model.
func (ctx *Context) EmbedText(text string) ([]float32, error) {
	switch {
	case ctx.handle == 0 || ctx.parent.handle == 0:
		return nil, fmt.Errorf("context is not initialized")
	case ctx.parent.n_embd <= 0:
		return nil, fmt.Errorf("model does not support embedding")
	}

	out := make([]float32, ctx.parent.n_embd)
	tok := uint32(0)
	ret := embed_text(ctx.handle, text, out, &tok)
	ctx.tokens.Add(uint64(tok))
	switch ret {
	case 0:
		return out, nil
	case 1:
		return nil, fmt.Errorf("number of tokens (%d) exceeds batch size", tok)
	case 2:
		return nil, fmt.Errorf("last token in the prompt is not SEP")
	case 3:
		return nil, fmt.Errorf("failed to decode/encode text")
	default:
		return nil, fmt.Errorf("failed to embed text (code=%d)", ret)
	}
}

func (ctx *Context) CompleteText(text string, n int) (string, error) {
	switch {
	case ctx.handle == 0 || ctx.parent.handle == 0:
		return "", fmt.Errorf("context is not initialized")
	}

	// align to the closest page of 128
	out := make([]byte, (n+511)&^511)
	ret := complete_text(ctx.handle, text, out, uint32(len(out)), uint32(n))
	switch ret {
	case 0:
		return string(out), nil
	case 1:
		return "", fmt.Errorf("failed to complete text")
	case 2:
		return "", fmt.Errorf("failed to evaluate initial prompt")
	case 3:
		return "", fmt.Errorf("failed to generate text")
	case 4:
		return "", fmt.Errorf("output buffer is too small (%d)", len(out))
	default:
		return "", fmt.Errorf("failed to complete text (code=%d)", ret)
	}
}
