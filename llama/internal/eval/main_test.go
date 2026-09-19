//go:build integration

package main

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestEmbeddingsQuality tests the embedding quality using the SICK dataset
func TestEmbeddingsQuality(t *testing.T) {
	data, err := loadSICK()
	require.NoError(t, err)

	// Create slices to store predicted and human scores
	embedScores := make([]float64, 0, len(data))
	humanScores := make([]float64, 0, len(data))

	// Load your language model
	m := loadModel()
	defer m.Close()

	// Embed the sentences and calculate similarities
	for _, v := range data {
		embeddingA, err := m.EmbedText(context.Background(), v.Pair[0])
		require.NoError(t, err)

		embeddingB, err := m.EmbedText(context.Background(), v.Pair[1])
		require.NoError(t, err)

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
	assert.True(t, spearman > 0.7, "Correlation is below acceptable threshold")
}

func TestMetrics(t *testing.T) {
	x := []float64{1, 2, 3}
	assert.Equal(t, []int{1, 2, 0}, argsort([]float64{30, 10, 20}))
	assert.Equal(t, []float64{3, 1, 2}, rank([]float64{30, 10, 20}))
	assert.Equal(t, float64(1), spearman(x, []float64{10, 20, 30}))
	assert.Equal(t, float64(-1), spearman(x, []float64{30, 20, 10}))
	assert.InDelta(t, 1, pearson(x, []float64{10, 20, 30}), 1e-6)
	assert.InDelta(t, -1, pearson(x, []float64{30, 20, 10}), 1e-6)
	assert.Zero(t, pearson(x, []float64{2, 2, 2}))
	assert.Zero(t, cosine([]float32{0, 0}, []float32{1, 2}))
	assert.InDelta(t, -1, cosine([]float32{3, 4}, []float32{-3, -4}), 1e-6)
	assert.InDelta(t, 3, cosineScaled([]float32{1, 0}, []float32{0, 1}, 4, 0), 1e-6)
	assert.InDelta(t, 2, mse(x, []float64{2, 4, 2}), 1e-6)
}
