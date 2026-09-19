package search_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/kelindar/search"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCodecFailures(t *testing.T) {
	for _, binary := range []bool{false, true} {
		t.Run(fmt.Sprint(binary), func(t *testing.T) {
			var encoded bytes.Buffer
			strings := search.NewIndex[string]()
			strings.Add([]float32{1, 0}, "cat")
			blobs := search.NewIndex[[]byte]()
			blobs.Add([]float32{1, 0}, []byte("cat"))
			write := strings.WriteTo
			read := strings.ReadFrom
			if binary {
				write, read = blobs.WriteTo, blobs.ReadFrom
			}
			n, err := write(&encoded)
			require.NoError(t, err)
			assert.Equal(t, int64(encoded.Len()), n)
			for cutoff := 0; cutoff < encoded.Len(); cutoff++ {
				_, err := read(bytes.NewReader(encoded.Bytes()[:cutoff]))
				assert.Error(t, err, "truncated at byte %d", cutoff)
			}
			_, err = read(bytes.NewReader(encoded.Bytes()))
			require.NoError(t, err)
			failure := errors.New("write failed")
			for cutoff := 0; cutoff < encoded.Len(); cutoff++ {
				_, err := write(&limitedWriter{remaining: cutoff, err: failure})
				assert.ErrorIs(t, err, failure, "writer failed at byte %d", cutoff)
			}
		})
	}
	index := search.NewIndex[string]()
	_, err := index.ReadFrom(bytes.NewReader([]byte{2}))
	assert.EqualError(t, err, "unsupported version: 2")
	path := filepath.Join(t.TempDir(), "missing", "index.bin")
	assert.Error(t, index.ReadFile(path))
	assert.Error(t, index.WriteFile(path))
	path = filepath.Join(t.TempDir(), "corrupt.bin")
	require.NoError(t, os.WriteFile(path, []byte("invalid compressed index"), 0600))
	assert.Error(t, index.ReadFile(path))
}

type limitedWriter struct {
	remaining int
	err       error
}

func (w *limitedWriter) Write(p []byte) (int, error) {
	if len(p) > w.remaining {
		n := w.remaining
		w.remaining = 0
		return n, w.err
	}
	w.remaining -= len(p)
	return len(p), nil
}

func TestCodecVectors(t *testing.T) {
	input := search.NewIndex[int]()
	input.Add([]float32{1, 0}, 42)
	var encoded bytes.Buffer
	_, err := input.WriteTo(&encoded)
	require.NoError(t, err)
	output := search.NewIndex[int]()
	_, err = output.ReadFrom(&encoded)
	require.NoError(t, err)
	// Only string and byte slice values are persisted; other types retain vectors.
	assert.Equal(t, []search.Result[int]{{Value: 0, Relevance: 1}}, output.Search([]float32{1, 0}, 1))
}

// A custom provider needs no imports from llama or openai.
type customProvider struct{}

func (customProvider) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Replace this demonstration mapping with your embedding service.
	if text == "cat" {
		return []float32{1, 0}, nil
	}
	return []float32{0, 1}, nil
}

func ExampleIndex() {
	provider := customProvider{}
	index := search.NewIndex[string]()
	for _, text := range []string{"cat", "dog"} {
		vector, err := provider.EmbedText(context.Background(), text)
		if err != nil {
			panic(err)
		}
		index.Add(vector, text)
	}
	query, err := provider.EmbedText(context.Background(), "cat")
	if err != nil {
		panic(err)
	}
	fmt.Println(index.Search(query, 1)[0].Value)
	// Output: cat
}
