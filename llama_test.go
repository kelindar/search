// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package search

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
BenchmarkLLM/encode-24         	     400	   2641772 ns/op	        18.00 tok/s	    2024 B/op	      11 allocs/op
*/
func BenchmarkLLM(b *testing.B) {
	m := loadModel()
	defer m.Close()

	text := "This is a test sentence we are going to generate embeddings for."
	b.Run("encode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := m.EmbedText(text)
			assert.NoError(b, err)
		}

	})
}

func loadModel() *Vectorizer {
	mod, _ := filepath.Abs("dist/MiniLM-L6-v2.Q8_0.gguf")
	ctx, err := NewVectorizer(mod, 512)
	if err != nil {
		panic(err)
	}
	return ctx
}

func TestEmbedText(t *testing.T) {
	m := loadModel()
	defer m.Close()

	var sb strings.Builder
	for i := 0; i < 10; i++ {
		sb.WriteString("This is a test sentence we are going to generate embeddings for.\n")
	}

	out, err := m.EmbedText(sb.String())
	assert.NoError(t, err)
	assert.NotZero(t, len(out))
}
