package llm

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func BenchmarkLLM(b *testing.B) {
	ctx := loadModel()
	defer ctx.Close()

	// text := "This is a test sentence we are going to generate embeddings for."
	text := "Hello, world!"

	b.Run("encode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ctx.EmbedText(text)
			assert.NoError(b, err)
		}
	})
}

func loadModel() *Model {
	mod, _ := filepath.Abs("dist/MiniLM-L6-v2.Q4_K_M.gguf")
	ctx, err := New(mod, 512)
	if err != nil {
		panic(err)
	}
	return ctx
}
