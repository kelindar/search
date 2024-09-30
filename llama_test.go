package llm

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
BenchmarkLLM/encode-24         	     486	   2430962 ns/op	    1536 B/op	       1 allocs/op
*/
func BenchmarkLLM(b *testing.B) {
	ctx := loadModel()
	defer ctx.Close()

	text := "This is a test sentence we are going to generate embeddings for."
	b.Run("encode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ctx.EmbedText(text)
			assert.NoError(b, err)
		}
	})
}

func loadModel() *Model {
	mod, _ := filepath.Abs("dist/MiniLM-L6-v2.Q4_K_M.gguf")
	//mod, _ := filepath.Abs("dist/Llama-3.2-1B-Instruct-Q6_K_L.gguf")
	ctx, err := New(mod, 512)
	if err != nil {
		panic(err)
	}
	return ctx
}
