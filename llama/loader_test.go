// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package llama

import (
	"context"
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
