// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package llama

import (
	"context"
	"fmt"
	"io"
	"strings"
	"sync/atomic"
)

// Vectorizer represents a loaded embedding model. Embedding calls may run
// concurrently. Finish all calls and close explicit contexts before Close.
type Vectorizer struct {
	handle uintptr
	n_embd int32
	pool   *pool[*Context]
}

// New loads a GGUF model. gpuLayers is nonnegative; zero uses the CPU.
// The native library is initialized once, on the first valid call to New.
func New(modelPath string, gpuLayers int) (*Vectorizer, error) {
	switch {
	case strings.TrimSpace(modelPath) == "":
		return nil, fmt.Errorf("model path is empty")
	case gpuLayers < 0 || uint64(gpuLayers) > uint64(^uint32(0)):
		return nil, fmt.Errorf("GPU layers must fit an unsigned 32-bit integer")
	}
	if err := initialize(); err != nil {
		return nil, err
	}
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

// Close releases the model and pooled contexts. Repeated calls are harmless.
// Embedding after Close returns an error. Close must not race with any model use.
func (m *Vectorizer) Close() error {
	if m.handle == 0 {
		return nil
	}
	m.pool.Close()
	free_model(m.handle)
	m.handle = 0
	return nil
}

// Context creates a caller-owned context of the given size, or an unusable
// context if allocation fails. Zero uses the model default. A Context must
// not be used concurrently and must be closed before its model.
func (m *Vectorizer) Context(size int) *Context {
	if m.handle == 0 || size < 0 || uint64(size) > uint64(^uint32(0)) {
		return &Context{parent: m}
	}
	return &Context{
		parent: m,
		handle: load_context(m.handle, uint32(size), true),
	}
}

// EmbedText returns a caller-owned vector. Cancellation cannot interrupt an
// in-flight native call, but is checked before and after it.
func (m *Vectorizer) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	switch {
	case strings.TrimSpace(text) == "":
		return nil, fmt.Errorf("text is empty")
	case m.handle == 0:
		return nil, fmt.Errorf("model is closed")
	}
	c := m.pool.Get()
	defer m.pool.Put(c)
	return c.EmbedText(ctx, text)
}

// EmbedBatch embeds texts sequentially, returning caller-owned vectors in input
// order. An empty batch does no work. Failure returns no partial results.
func (m *Vectorizer) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	for i, text := range texts {
		if strings.TrimSpace(text) == "" {
			return nil, fmt.Errorf("text %d is empty", i)
		}
	}
	out := make([][]float32, len(texts))
	for i, text := range texts {
		v, err := m.EmbedText(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("embed text %d: %w", i, err)
		}
		out[i] = v
	}
	return out, nil
}

// --------------------------------- Context ---------------------------------

// Context represents a context for embedding text using the model.
type Context struct {
	parent *Vectorizer
	handle uintptr
	tokens atomic.Uint64
}

// Close releases the context. Repeated calls are harmless; embedding after Close
// returns an error. Close must not race with embedding on this context.
func (ctx *Context) Close() error {
	if ctx.handle == 0 {
		return nil
	}
	free_context(ctx.handle)
	ctx.handle = 0
	return nil
}

// Tokens returns the number of tokens processed by the context.
func (ctx *Context) Tokens() uint {
	return uint(ctx.tokens.Load())
}

// EmbedText returns a caller-owned vector. Cancellation cannot interrupt native
// inference. Tokens counts native work even if cancellation discards the result.
func (ctx *Context) EmbedText(call context.Context, text string) ([]float32, error) {
	if err := call.Err(); err != nil {
		return nil, err
	}
	switch {
	case strings.TrimSpace(text) == "":
		return nil, fmt.Errorf("text is empty")
	case ctx.handle == 0 || ctx.parent == nil || ctx.parent.handle == 0:
		return nil, fmt.Errorf("context is not initialized")
	case ctx.parent.n_embd <= 0:
		return nil, fmt.Errorf("model does not support embedding")
	}

	out := make([]float32, ctx.parent.n_embd)
	tok := uint32(0)
	ret := embed_text(ctx.handle, text, out, &tok)
	ctx.tokens.Add(uint64(tok))
	if err := call.Err(); err != nil {
		return nil, err
	}
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
