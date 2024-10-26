package main

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestEmbeddingsQuality tests the embedding quality using the SICK dataset
func TestEmbeddingsQuality(t *testing.T) {
	data, err := loadSICK()
	assert.NoError(t, err)

	// Create slices to store predicted and human scores
	embedScores := make([]float64, 0, len(data))
	humanScores := make([]float64, 0, len(data))

	// Load your language model
	llm := loadModel()
	defer llm.Close()

	// Embed the sentences and calculate similarities
	for _, v := range data {
		embeddingA, err := llm.EmbedText(v.Pair[0])
		assert.NoError(t, err)

		embeddingB, err := llm.EmbedText(v.Pair[1])
		assert.NoError(t, err)

		// Calculate similarity (you can replace CosineSimilarity with your own method)
		similarity := cosineScaled(embeddingA, embeddingB, 3.85, 0.5)

		// Clamp the similarity to 0 or 1

		embedScores = append(embedScores, similarity)
		humanScores = append(humanScores, v.Rank)

		// Print each comparison for debugging (optional)
		//fmt.Printf(" - \"%s\" vs \"%s\"\n", v.Pair[0], v.Pair[1])
		//fmt.Printf("   Human: %.2f, Predicted: %.2f\n", v.Rank, similarity)

	}

	// Calculate correlations between human scores and predicted scores
	pearson := pearson(humanScores, embedScores)
	spearman := spearman(humanScores, embedScores)
	mse := mse(humanScores, embedScores)

	fmt.Printf("Spearman correlation between human scores and predicted scores: %.4f\n", spearman)
	fmt.Printf("Pearson correlation between human scores and predicted scores: %.4f\n", pearson)
	fmt.Printf("Mean Squared Error between human scores and predicted scores: %.4f\n", mse)

	// Assert that the correlation meets your desired threshold
	assert.False(t, spearman > 0.1, "Correlation is below acceptable threshold")
}
