package main

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/kelindar/llm"
	"github.com/kelindar/llm/search"
)

func main() {
	m, err := llm.New("../dist/MiniLM-L6-v2.Q8_0.gguf", 0)
	if err != nil {
		panic(err)
	}

	defer m.Close()

	// Load a pre-embedded dataset and create an exact search index
	data, _ := loadDataset("../search/dataset.gob")
	index := search.NewIndex[string]()

	// Embed the sentences and calculate similarities
	for _, v := range data {
		index.Add(v.Vector, v.Pair[0]) // use m.EmbedText() for real-time embedding
	}

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
			for relevance, text := range results {
				switch {
				case relevance >= 0.85:
					fmt.Printf(" ✅ %s (%.0f%%)\n", text, math.Round(relevance*100))
				case relevance >= 0.5:
					fmt.Printf(" ❔ %s (%.0f%%)\n", text, math.Round(relevance*100))
				default:
					fmt.Printf(" ❌ %s (%.0f%%)\n", text, math.Round(relevance*100))
				}
			}
		}
	}
}

type record struct {
	Pair   [2]string `gob:"pair"`
	Rank   float64   `gob:"rank"`
	Label  string    `gob:"label"`
	Vector []float32 `gob:"vector"`
}

func loadDataset(path string) ([]record, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var data []record
	r := gob.NewDecoder(file)
	if err := r.Decode(&data); err != nil {
		return nil, err
	}

	return data, nil
}

/*
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
*/
