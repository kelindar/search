// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package llama

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadFailure(t *testing.T) {
	if os.Getenv("SEARCH_TEST_BAD_LIBRARY") == "1" {
		_, err := New("unused.gguf", 0)
		require.ErrorContains(t, err, "load llama library")
		_, again := New("unused.gguf", 0)
		assert.Equal(t, err, again)
		return
	}

	// Use a fresh process so the once-only loader cannot affect other tests.
	dir := t.TempDir()
	name, env := "libllama_go.so", "LD_LIBRARY_PATH"
	switch runtime.GOOS {
	case "windows":
		name, env = "llama_go.dll", "PATH"
	case "darwin":
		name, env = "libllama_go.dylib", "DYLD_LIBRARY_PATH"
	}
	require.NoError(t, os.WriteFile(filepath.Join(dir, name), []byte("invalid library"), 0600))
	t.Setenv(env, dir)
	t.Setenv("SEARCH_TEST_BAD_LIBRARY", "1")
	command := exec.Command(os.Args[0], "-test.run=^TestLoadFailure$")
	output, err := command.CombinedOutput()
	require.NoError(t, err, "%s", output)
}

func TestLibDirs_Windows(t *testing.T) {
	ext, libs := findLibDirs("windows")
	assert.Equal(t, ".dll", ext)
	assert.NotEmpty(t, libs)
}

func TestLibDirs_Darwin(t *testing.T) {
	ext, libs := findLibDirs("darwin")
	assert.Equal(t, ".dylib", ext)
	assert.NotEmpty(t, libs)
}

func TestLibDirs_Linux(t *testing.T) {
	ext, libs := findLibDirs("linux")
	assert.Equal(t, ".so", ext)
	assert.NotEmpty(t, libs)
}

func TestFindLibrary_Err(t *testing.T) {
	_, err := findLibrary("nonexistent", "linux")
	assert.Error(t, err)
}

func TestValidation(t *testing.T) {
	for _, tc := range []struct {
		path   string
		layers int
	}{{"", 0}, {"model.gguf", -1}} {
		m, err := New(tc.path, tc.layers)
		assert.Error(t, err)
		assert.Nil(t, m)
	}
	m := &Vectorizer{}
	_, err := m.EmbedText(context.Background(), "hello")
	assert.ErrorContains(t, err, "closed")
	_, err = m.EmbedText(context.Background(), " ")
	assert.ErrorContains(t, err, "empty")
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = m.EmbedText(canceled, "hello")
	assert.ErrorIs(t, err, context.Canceled)
	batch, err := m.EmbedBatch(context.Background(), nil)
	require.NoError(t, err)
	assert.Empty(t, batch)
	batch, err = m.EmbedBatch(context.Background(), []string{"hello", ""})
	assert.ErrorContains(t, err, "text 1 is empty")
	assert.Nil(t, batch)
	require.NoError(t, m.Close())
	c := m.Context(0)
	_, err = c.EmbedText(context.Background(), "hello")
	assert.ErrorContains(t, err, "not initialized")
	require.NoError(t, c.Close())
}

func TestNewFailures(t *testing.T) {
	originalInitialize := initialize
	originalLoadModel := load_model
	t.Cleanup(func() {
		initialize = originalInitialize
		load_model = originalLoadModel
	})

	initErr := errors.New("initialize failed")
	initialize = func() error { return initErr }
	model, err := New("model.gguf", 0)
	assert.ErrorIs(t, err, initErr)
	assert.Nil(t, model)

	initialize = func() error { return nil }
	load_model = func(string, uint32) uintptr { return 0 }
	model, err = New("model.gguf", 0)
	assert.ErrorContains(t, err, "failed to load model")
	assert.Nil(t, model)
}

func TestBatchError(t *testing.T) {
	originalEmbedText := embed_text
	originalFreeContext := free_context
	originalFreeModel := free_model
	t.Cleanup(func() {
		embed_text = originalEmbedText
		free_context = originalFreeContext
		free_model = originalFreeModel
	})

	calls := 0
	embed_text = func(_ uintptr, _ string, output []float32, tokens *uint32) int {
		calls++
		*tokens = uint32(calls)
		output[0] = 1
		if calls == 2 {
			return 3
		}
		return 0
	}
	free_context = func(uintptr) {}
	free_model = func(uintptr) {}

	model := &Vectorizer{handle: 1, n_embd: 1}
	model.pool = newPool(1, func() *Context {
		return &Context{parent: model, handle: 1}
	})
	defer model.Close()

	vectors, err := model.EmbedBatch(context.Background(), []string{"one", "two"})
	assert.ErrorContains(t, err, "embed text 1")
	assert.Nil(t, vectors)
	assert.Equal(t, 2, calls)
}

func TestContextErrors(t *testing.T) {
	cases := []struct {
		name string
		ctx  *Context
		text string
		want string
	}{
		{name: "empty", ctx: &Context{parent: &Vectorizer{handle: 1, n_embd: 1}, handle: 1}, text: " ", want: "text is empty"},
		{name: "no handle", ctx: &Context{parent: &Vectorizer{handle: 1, n_embd: 1}}, text: "hello", want: "not initialized"},
		{name: "no parent", ctx: &Context{handle: 1}, text: "hello", want: "not initialized"},
		{name: "closed model", ctx: &Context{parent: &Vectorizer{}, handle: 1}, text: "hello", want: "not initialized"},
		{name: "no embedding", ctx: &Context{parent: &Vectorizer{handle: 1}, handle: 1}, text: "hello", want: "model does not support embedding"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			output, err := test.ctx.EmbedText(context.Background(), test.text)
			assert.ErrorContains(t, err, test.want)
			assert.Nil(t, output)
		})
	}
}

func TestContextNativeErrors(t *testing.T) {
	originalEmbedText := embed_text
	t.Cleanup(func() { embed_text = originalEmbedText })

	code := 0
	embed_text = func(_ uintptr, _ string, output []float32, tokens *uint32) int {
		*tokens = 4
		output[0] = 1
		output[1] = 2
		return code
	}
	parent := &Vectorizer{handle: 1, n_embd: 2}
	ctx := &Context{parent: parent, handle: 1}
	for _, test := range []struct {
		name string
		code int
		want string
	}{
		{name: "tokens", code: 1, want: "exceeds batch size"},
		{name: "separator", code: 2, want: "last token"},
		{name: "decode", code: 3, want: "decode/encode"},
		{name: "unknown", code: 99, want: "code=99"},
	} {
		t.Run(test.name, func(t *testing.T) {
			code = test.code
			output, err := ctx.EmbedText(context.Background(), "hello")
			assert.ErrorContains(t, err, test.want)
			assert.Nil(t, output)
		})
	}

	code = 0
	output, err := ctx.EmbedText(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, []float32{1, 2}, output)
	assert.Equal(t, uint(20), ctx.Tokens())

	canceled, cancel := context.WithCancel(context.Background())
	embed_text = func(_ uintptr, _ string, output []float32, tokens *uint32) int {
		*tokens = 5
		output[0] = 1
		output[1] = 2
		cancel()
		return 0
	}
	output, err = ctx.EmbedText(canceled, "hello")
	assert.ErrorIs(t, err, context.Canceled)
	assert.Nil(t, output)
	assert.Equal(t, uint(25), ctx.Tokens())
}

func TestContextClose(t *testing.T) {
	originalFreeContext := free_context
	t.Cleanup(func() { free_context = originalFreeContext })

	var freed []uintptr
	free_context = func(handle uintptr) { freed = append(freed, handle) }
	ctx := &Context{handle: 42}
	require.NoError(t, ctx.Close())
	require.NoError(t, ctx.Close())
	assert.Equal(t, uintptr(0), ctx.handle)
	assert.Equal(t, []uintptr{42}, freed)
}

func TestPoolLifecycle(t *testing.T) {
	pool := newPool(1, func() *testResource { return &testResource{} })
	first := pool.Get()
	second := pool.Get()
	pool.Put(first)
	pool.Put(second)
	assert.False(t, first.closed)
	assert.True(t, second.closed)

	pool.Close()
	assert.True(t, first.closed)
}

type testResource struct {
	closed bool
}

func (r *testResource) Close() error {
	r.closed = true
	return nil
}
