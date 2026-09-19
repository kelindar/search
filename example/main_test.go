package main

import (
	"path/filepath"
	"testing"

	"github.com/kelindar/search"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadIndex(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.bin")
	index := search.NewIndex[string]()
	index.Add([]float32{1, 0}, "cat")
	index.Add([]float32{0, 1}, "dog")
	require.NoError(t, index.WriteFile(path))
	loaded := loadIndex(path)
	assert.Equal(t, 2, loaded.Len())
	results := loaded.Search([]float32{1, 0}, 2)
	require.Len(t, results, 2)
	assert.Equal(t, "cat", results[0].Value)
	assert.InDelta(t, 1, results[0].Relevance, 1e-6)
}
