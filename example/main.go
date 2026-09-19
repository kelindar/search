package main

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/kelindar/search"
	"github.com/kelindar/search/llama"
)

func main() {
	m, err := llama.New("../dist/MiniLM-L6-v2.Q8_0.gguf", 0)
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
			embedding, err := m.EmbedText(context.Background(), query)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				continue
			}

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
