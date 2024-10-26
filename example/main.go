package main

import (
	"fmt"

	"github.com/kelindar/llm"
	"github.com/kelindar/llm/search"
)

func main() {
	m, err := llm.New("../dist/MiniLM-L6-v2.Q8_0.gguf", 0)
	if err != nil {
		panic(err)
	}

	defer m.Close()

	prompts := []string{
		"A boy is studying a calendar",
		"A boy is staring at a calendar",
		"A man is making a sketch",
		"A man is drawing",
	}

	embeddings := make([][]float32, len(prompts))
	for i, prompt := range prompts {
		embeddings[i], err = m.EmbedText(prompt)
		if err != nil {
			panic(err)
		}
	}

	// Compute pairwise cosine similarities and print them out
	for i := 0; i < len(embeddings); i++ {
		for j := i + 1; j < len(embeddings); j++ {
			cos := search.Cosine(embeddings[i], embeddings[j])
			fmt.Printf("\n * Similarity = %.2f\n", cos)
			fmt.Printf("   1: %s\n", prompts[i])
			fmt.Printf("   2: %s\n", prompts[j])
		}
	}
}
