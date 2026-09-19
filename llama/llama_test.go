//go:build integration

// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package llama

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

/*
BenchmarkLLM/encode-24         	     465	   2305573 ns/op	    2024 B/op	      11 allocs/op
*/
func BenchmarkLLM(b *testing.B) {
	m := loadModel(b)
	defer m.Close()

	text := "This is a test sentence we are going to generate embeddings for."
	b.Run("encode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := m.EmbedText(context.Background(), text)
			assert.NoError(b, err)
		}

	})
}

func loadModel(t testing.TB) *Vectorizer {
	t.Helper()
	library, err := findLlama()
	require.NoError(t, err)
	t.Logf("native library: %s", library)
	mod, _ := filepath.Abs("../dist/MiniLM-L6-v2.Q8_0.gguf")
	ctx, err := New(mod, 0)
	require.NoError(t, err)
	return ctx
}

func TestEmbedText(t *testing.T) {
	m := loadModel(t)
	defer m.Close()

	var sb strings.Builder
	for i := 0; i < 10; i++ {
		sb.WriteString("This is a test sentence we are going to generate embeddings for.\n")
	}

	out, err := m.EmbedText(context.Background(), sb.String())
	assert.NoError(t, err)
	assert.NotZero(t, len(out))

	t.Run("batch and ownership", func(t *testing.T) {
		texts := []string{"A cat sleeps.", "A dog runs."}
		vectors, err := m.EmbedBatch(context.Background(), texts)
		require.NoError(t, err)
		require.Len(t, vectors, 2)
		for i, text := range texts {
			want, err := m.EmbedText(context.Background(), text)
			require.NoError(t, err)
			assert.InDeltaSlice(t, want, vectors[i], 1e-5)
		}
		first := append([]float32(nil), vectors[0]...)
		vectors[1][0]++
		assert.Equal(t, first, vectors[0])
	})

	t.Run("explicit context", func(t *testing.T) {
		c := m.Context(0)
		defer c.Close()
		_, err := c.EmbedText(context.Background(), "A cat sleeps.")
		require.NoError(t, err)
		assert.Positive(t, c.Tokens())
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		out, err := c.EmbedText(ctx, "A dog runs.")
		assert.ErrorIs(t, err, context.Canceled)
		assert.Nil(t, out)
	})
}
