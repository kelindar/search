package llm

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
)

type entry struct {
	Pair [2]string
	Rank float64
}

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
		similarity := cosineScaled(embeddingA, embeddingB, 5, 0.5)
		embedScores = append(embedScores, similarity)
		humanScores = append(humanScores, v.Rank)

		// Print each comparison for debugging (optional)
		fmt.Printf(" - \"%s\" vs \"%s\"\n", v.Pair[0], v.Pair[1])
		fmt.Printf("   Human score: %.2f, Predicted: %.2f\n\n", v.Rank, similarity)
	}

	// Calculate correlations between human scores and predicted scores
	pearson := pearson(humanScores, embedScores)
	spearman := spearman(humanScores, embedScores)
	mse := calculateMSE(humanScores, embedScores)

	fmt.Printf("Spearman correlation between human scores and predicted scores: %.4f\n", spearman)
	fmt.Printf("Pearson correlation between human scores and predicted scores: %.4f\n", pearson)
	fmt.Printf("Mean Squared Error between human scores and predicted scores: %.4f\n", mse)

	// Assert that the correlation meets your desired threshold
	assert.False(t, spearman > 0.7, "Correlation is below acceptable threshold")
}

func TestGridSearch(t *testing.T) {
	data, err := loadSICK()
	assert.NoError(t, err)

	// Define grid search parameters for k and bias
	kValues := []float64{1.0, 3.0, 5.0, 7.0, 10.0}
	biasValues := []float64{-0.5, 0.0, 0.5, 1.0}

	// Track the best combination of parameters
	var bestK, bestBias float64
	bestPearson, bestSpearman, bestMSE := -1.0, -1.0, math.MaxFloat64

	// Load your language model
	llm := loadModel()
	defer llm.Close()

	for _, k := range kValues {
		for _, bias := range biasValues {
			embedScores := make([]float64, 0, len(data))
			humanScores := make([]float64, 0, len(data))

			// Embed the sentences and calculate similarities with the current k and bias
			for _, v := range data {
				embeddingA, err := llm.EmbedText(v.Pair[0])
				assert.NoError(t, err)

				embeddingB, err := llm.EmbedText(v.Pair[1])
				assert.NoError(t, err)

				// Calculate similarity using the current k and bias
				similarity := cosineScaled(embeddingA, embeddingB, k, bias)
				embedScores = append(embedScores, similarity)
				humanScores = append(humanScores, v.Rank)
			}

			// Calculate correlations and MSE
			pearson := pearson(humanScores, embedScores)
			spearman := spearman(humanScores, embedScores)
			mse := calculateMSE(humanScores, embedScores)

			// Print the results for this combination
			fmt.Printf("k: %.2f, bias: %.2f -> Spearman: %.4f, Pearson: %.4f, MSE: %.4f\n", k, bias, spearman, pearson, mse)

			// Check if this combination is the best so far
			if pearson > bestPearson || (pearson == bestPearson && mse < bestMSE) {
				bestK = k
				bestBias = bias
				bestPearson = pearson
				bestSpearman = spearman
				bestMSE = mse
			}
		}
	}

	// Print the best combination of k and bias
	fmt.Printf("Best k: %.2f, Best bias: %.2f -> Best Spearman: %.4f, Best Pearson: %.4f, Best MSE: %.4f\n", bestK, bestBias, bestSpearman, bestPearson, bestMSE)

	// Assert that the best combination meets the desired threshold
	assert.True(t, bestSpearman > 0.7, "Spearman correlation is below acceptable threshold")
	assert.FailNow(t, "Test not implemented")
}

// loadSICK parses the SICK CSV dataset and returns sentence pairs with their relatedness scores
func loadSICK() ([]entry, error) {
	file, err := os.Open("dist/SICK_dataset.txt")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = '\t'    // Tab-separated file
	_, err = reader.Read() // Skip header line
	if err != nil {
		return nil, err
	}

	out := make([]entry, 0, 4600)
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}

		sentenceA := record[1]
		sentenceB := record[2]
		relatednessScore, err := strconv.ParseFloat(record[3], 64)
		if err != nil {
			return nil, err
		}

		out = append(out, entry{Pair: [2]string{sentenceA, sentenceB}, Rank: relatednessScore})
	}

	return out, nil
}

// rank calculates the ranks of the elements in the array
func rank(data []float64) []float64 {
	n := len(data)
	rankArray := make([]float64, n)
	sortedIndices := argsort(data)

	// Assign ranks
	for i, idx := range sortedIndices {
		rankArray[idx] = float64(i + 1)
	}

	return rankArray
}

// argsort returns the indices of the sorted array
func argsort(data []float64) []int {
	type kv struct {
		Index int
		Value float64
	}

	var sortedData []kv
	for i, v := range data {
		sortedData = append(sortedData, kv{i, v})
	}

	// Sort based on value
	sort.Slice(sortedData, func(i, j int) bool {
		return sortedData[i].Value < sortedData[j].Value
	})

	// Extract sorted indices
	indices := make([]int, len(data))
	for i, kv := range sortedData {
		indices[i] = kv.Index
	}

	return indices
}

// spearman computes the Spearman rank correlation coefficient between two sets of scores
func spearman(humanScores, predictedScores []float64) float64 {
	if len(humanScores) != len(predictedScores) {
		log.Fatalf("Both score sets must have the same length")
	}

	// Compute rank arrays
	humanRanks := rank(humanScores)
	predictedRanks := rank(predictedScores)

	// Calculate Spearman correlation
	n := float64(len(humanScores))
	var sumDiffSquared float64
	for i := range humanScores {
		diff := humanRanks[i] - predictedRanks[i]
		sumDiffSquared += diff * diff
	}

	return 1 - (6*sumDiffSquared)/(n*(n*n-1))
}

// cosine calculates the cosine similarity between two vectors
func cosine(vec1, vec2 []float32) float64 {
	if len(vec1) != len(vec2) {
		log.Fatalf("Vectors must be of same length")
	}
	var dotProduct, normA, normB float64
	for i := range vec1 {
		dotProduct += float64(vec1[i] * vec2[i])
		normA += float64(vec1[i] * vec1[i])
		normB += float64(vec2[i] * vec2[i])
	}
	if normA == 0 || normB == 0 {
		return 0.0
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func cosineScaled(vec1, vec2 []float32, k, bias float64) float64 {
	similarity := cosine(vec1, vec2)
	return 4/(1+math.Exp(-k*(similarity-bias))) + 1
}

// pearson calculates the Pearson correlation coefficient between two sets of scores
func pearson(x, y []float64) float64 {
	if len(x) != len(y) {
		log.Fatalf("Both score sets must have the same length")
	}

	n := float64(len(x))
	var sumX, sumY, sumXY, sumX2, sumY2 float64

	for i := range x {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}

	numerator := n*sumXY - sumX*sumY
	denominator := math.Sqrt((n*sumX2 - sumX*sumX) * (n*sumY2 - sumY*sumY))

	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}

// calculateMSE calculates the mean squared error between two sets of scores
func calculateMSE(humanScores, predictedScores []float64) float64 {
	if len(humanScores) != len(predictedScores) {
		log.Fatalf("Both score sets must have the same length")
	}

	var sumSquaredError float64
	for i := range humanScores {
		error := humanScores[i] - predictedScores[i]
		sumSquaredError += error * error
	}

	return sumSquaredError / float64(len(humanScores))
}
