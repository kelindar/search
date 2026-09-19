package search_test

import (
	"context"
	"fmt"

	"github.com/kelindar/search"
)

// A custom provider needs no imports from llama or openai.
type customProvider struct{}

func (customProvider) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Replace this demonstration mapping with your embedding service.
	if text == "cat" {
		return []float32{1, 0}, nil
	}
	return []float32{0, 1}, nil
}

func ExampleIndex() {
	provider := customProvider{}
	index := search.NewIndex[string]()
	for _, text := range []string{"cat", "dog"} {
		vector, err := provider.EmbedText(context.Background(), text)
		if err != nil {
			panic(err)
		}
		index.Add(vector, text)
	}
	query, err := provider.EmbedText(context.Background(), "cat")
	if err != nil {
		panic(err)
	}
	fmt.Println(index.Search(query, 1)[0].Value)
	// Output: cat
}
