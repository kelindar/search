// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package search

import (
	"fmt"
	"math/rand/v2"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
cpu: 13th Gen Intel(R) Core(TM) i7-13700K
BenchmarkIndex/search-24         	   10000	    102210 ns/op	     160 B/op	       3 allocs/op
*/
func BenchmarkIndex(b *testing.B) {
	index := loadIndex(b)

	b.Run("search", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = index.Search(index.arr[i%1000].Vector, 5)
		}
	})
}

func TestIndex(t *testing.T) {
	index := loadIndex(t)
	assert.Equal(t, 4802, index.Len())

	for i := 0; i < 1000; i++ {
		record := index.arr[i]

		results := index.Search(record.Vector, 5)
		assert.Equal(t, 5, len(results))
		assert.InDelta(t, 1, results[0].Relevance, 1e-4)
	}
}

func TestCodec_String(t *testing.T) {
	const name = "test.bin"

	// Create an index
	input := NewIndex[string]()
	for i := 0; i < 10; i++ {
		input.Add(randVec(), fmt.Sprintf("item-%d", i))
	}

	// Marshal, unmarshal and compare
	assert.NoError(t, input.WriteFile(name))
	defer os.Remove(name)
	output := NewIndex[string]()
	assert.NoError(t, output.ReadFile(name))
	assert.Equal(t, input, output)
}

func TestCodec_Binary(t *testing.T) {
	const name = "test.bin"

	// Create an index
	input := NewIndex[[]byte]()
	for i := 0; i < 10; i++ {
		input.Add(randVec(), []byte(fmt.Sprintf("item-%d", i)))
	}

	// Marshal, unmarshal and compare
	assert.NoError(t, input.WriteFile(name))
	defer os.Remove(name)
	output := NewIndex[[]byte]()
	assert.NoError(t, output.ReadFile(name))
	assert.Equal(t, input, output)
}

type record struct {
	Text   string
	Vector []float32
}

func loadIndex(t testing.TB) *Index[string] {
	index := NewIndex[string]()
	assert.NoError(t, index.ReadFile("dist/dataset.bin"))
	return index
}

func randVec() []float32 {
	v := make([]float32, 384)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}
