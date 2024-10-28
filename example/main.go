package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/kelindar/search"
)

func main() {
	m, err := search.NewVectorizer("../dist/MiniLM-L6-v2.Q8_0.gguf", 0)
	if err != nil {
		panic(err)
	}

	defer m.Close()

	// Load a pre-embedded dataset and create an exact search index
	index := loadIndex("../dist/dataset.bin")

	r := bufio.NewReader(os.Stdin)
	for {
		fmt.Printf("Enter a sentence to search (or 'exit' to quit): ")
		query, _ := r.ReadString('\n')
		query = strings.TrimSpace(query)

		switch q := strings.TrimSpace(query); q {
		case "exit", "quit", "q", "bye", "":
			return
		default:

			// Embed the query
			embedding, _ := m.EmbedText(query)

			// Perform the search query
			start := time.Now()
			results := index.Search(embedding, 10)

			// Print the results
			fmt.Printf("results found (elapsed=%v) :\n", time.Since(start))
			for _, r := range results {
				switch {
				case r.Relevance >= 0.85:
					fmt.Printf(" ✅ %s (%.0f%%)\n", r.Value, math.Round(r.Relevance*100))
				case r.Relevance >= 0.5:
					fmt.Printf(" ❔ %s (%.0f%%)\n", r.Value, math.Round(r.Relevance*100))
				default:
					fmt.Printf(" ❌ %s (%.0f%%)\n", r.Value, math.Round(r.Relevance*100))
				}
			}
		}
	}
}

func loadIndex(path string) *search.Index[string] {
	index := search.NewIndex[string]()
	if err := index.ReadFile(path); err != nil {
		panic(err)
	}
	return index
}

/*
func main() {
	m, err := search.New("../dist/MiniLM-L6-v2.Q8_0.gguf", 0)
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
*/
