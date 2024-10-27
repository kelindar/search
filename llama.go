package search

import (
	"fmt"
	"io"
	"sync/atomic"
)

// Vectorizer represents a loaded LLM/Embedding model.
type Vectorizer struct {
	handle uintptr
	n_embd int32
	pool   *pool[*Context]
}

// NewVectorizer creates a new vectorizer model from the given model file.
func NewVectorizer(modelPath string, gpuLayers int) (*Vectorizer, error) {
	handle := load_model(modelPath, uint32(gpuLayers))
	if handle == 0 {
		return nil, fmt.Errorf("failed to load model (%s)", modelPath)
	}

	model := &Vectorizer{
		handle: handle,
		n_embd: embed_size(handle),
	}

	// Initialize the context pool to reduce allocations
	model.pool = newPool(16, func() *Context {
		return model.Context(0)
	})
	return model, nil
}

// Close closes the model and releases any resources associated with it.
func (m *Vectorizer) Close() error {
	free_model(m.handle)
	m.handle = 0
	m.pool.Close()
	return nil
}

// Context creates a new context of the given size.
func (m *Vectorizer) Context(size int) *Context {
	return &Context{
		parent: m,
		handle: load_context(m.handle, uint32(size), true),
	}
}

// EmbedText embeds the given text using the model.
func (m *Vectorizer) EmbedText(text string) ([]float32, error) {
	ctx := m.pool.Get()
	defer m.pool.Put(ctx)
	return ctx.EmbedText(text)
}

// --------------------------------- Context ---------------------------------

// Context represents a context for embedding text using the model.
type Context struct {
	parent *Vectorizer
	handle uintptr
	tokens atomic.Uint64
}

// Close closes the context and releases any resources associated with it.
func (ctx *Context) Close() error {
	free_context(ctx.handle)
	ctx.handle = 0
	return nil
}

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

// --------------------------------- Resource Pool ---------------------------------

// Pool is a generic pool of resources that can be reused.
type pool[T io.Closer] struct {
	pool chan T
	make func() T
}

// newPool creates a new pool of resources.
func newPool[T io.Closer](size int, new func() T) *pool[T] {
	return &pool[T]{
		pool: make(chan T, size),
		make: new,
	}
}

// Get returns a resource from the pool or creates a new one.
func (p *pool[T]) Get() T {
	select {
	case x := <-p.pool:
		return x
	default:
		return p.make()
	}
}

// Put returns the resource to the pool.
func (p *pool[T]) Put(x T) {
	select {
	case p.pool <- x:
	default:
		x.Close() // Close the resource if the pool is full
	}
}

// Close closes the pool and releases any resources associated with it.
func (p *pool[T]) Close() {
	close(p.pool)
	for x := range p.pool {
		x.Close()
	}
}
