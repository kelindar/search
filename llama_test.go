package llm

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
	llm := loadModel()
	defer llm.Close()

	text := "This is a test sentence we are going to generate embeddings for."
	b.Run("encode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := llm.EmbedText(text)
			assert.NoError(b, err)
		}

	})
}

func loadModel() *Model {
	mod, _ := filepath.Abs("dist/MiniLM-L6-v2.Q4_K_M.gguf")
	//mod, _ := filepath.Abs("dist/Llama-3.2-1B-Instruct-Q6_K_L.gguf")
	//mod, _ := filepath.Abs("dist/nomic-embed-text-v1.Q4_K_M.gguf")
	//mod, _ := filepath.Abs("dist/snowflake-arctic-embed-m-long--Q4_K_M.GGUF")
	//mod, _ := filepath.Abs("dist/snowflake-arctic-embed-m-long--Q5_K_M.GGUF")
	//mod, _ := filepath.Abs("dist/e5-base-v2.Q5_K_M.gguf")
	ctx, err := New(mod, 512)
	if err != nil {
		panic(err)
	}
	return ctx
}

func TestEmbedText(t *testing.T) {
	llm := loadModel()
	defer llm.Close()

	var sb strings.Builder
	for i := 0; i < 10; i++ {
		sb.WriteString("This is a test sentence we are going to generate embeddings for.\n")
	}

	out, err := llm.EmbedText(sb.String())
	assert.NoError(t, err)
	assert.NotZero(t, len(out))
}
