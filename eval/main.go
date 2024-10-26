package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/kelindar/llm"
	"github.com/kelindar/llm/search"
)

func loadModel() *llm.Model {
	model := "../dist/MiniLM-L6-v2.Q8_0.gguf"
	fmt.Printf("Loading model: %s\n", model)

	mod, _ := filepath.Abs(model)
	ctx, err := llm.New(mod, 0)
	if err != nil {
		panic(err)
	}

	return ctx
}

func main() {
	// mainEval()
	//mainSearchHNSW()
	mainSearchBrute()
	//mainSearchLSH()
}

func mainEval() {
	data, _ := loadSICK()

	// Create slices to store predicted and human scores
	embedScores := make([]float64, 0, len(data))
	humanScores := make([]float64, 0, len(data))

	// Load your language model
	llm := loadModel()
	defer llm.Close()

	// Embed the sentences and calculate similarities
	start := time.Now()
	for _, v := range data {
		embeddingA, _ := llm.EmbedText(v.Pair[0])
		embeddingB, _ := llm.EmbedText(v.Pair[1])

		// Calculate similarity (you can replace CosineSimilarity with your own method)
		similarity := cosineScaled(embeddingA, embeddingB, 3.85, 0.5)

		// Clamp the similarity to 0 or 1

		embedScores = append(embedScores, similarity)
		humanScores = append(humanScores, v.Rank)

		// Print each comparison for debugging (optional)
		//fmt.Printf(" - \"%s\" vs \"%s\"\n", v.Pair[0], v.Pair[1])
		//fmt.Printf("   Human: %.2f, Predicted: %.2f\n", v.Rank, similarity)
	}

	elapsed := time.Since(start)
	count := len(data) * 2 // two sentences per comparison
	fmt.Printf("✅ Processed %d sentences in %s, at a rate of %s per sentence\n", count, elapsed, elapsed/time.Duration(count))

	// Calculate correlations between human scores and predicted scores
	pearson := pearson(humanScores, embedScores)
	spearman := spearman(humanScores, embedScores)
	mse := mse(humanScores, embedScores)

	if spearman < 0.7 {
		fmt.Printf("❌ Spearman correlation of %.4f is below acceptable threshold\n", spearman)
	} else {
		fmt.Printf("✅ Spearman correlation between human scores and predicted scores: %.4f\n", spearman)
	}

	fmt.Printf("✅ Pearson correlation: %.4f, MSE: %.4f\n", pearson, mse)
}

type embedding struct {
	Vector []float32
	Text   string
}

func mainSearchBrute() {

	// Load your language model
	llm := loadModel()
	defer llm.Close()

	g := search.NewBag[string](384)

	// Embed the sentences and calculate similarities
	data, _ := loadSICK()
	for _, v := range data {
		text := strings.TrimSpace(v.Pair[0])
		vector, _ := llm.EmbedText(text)
		g.Add(vector, text)
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
			embedding, _ := llm.EmbedText(query)

			start := time.Now()
			results := g.Search(embedding, 50)

			// Print the results
			fmt.Printf("found %d results (%v)\n", len(results), time.Since(start))
			for i, v := range results {
				printResult(embedding, v.Vector, v.Value)
				if i >= 10 {
					break
				}
			}
		}
	}
}

func mainSearchHNSW() {

	// Load your language model
	llm := loadModel()
	defer llm.Close()

	g := search.NewGraph[int]()
	b := search.NewBag[string](384)

	// Embed the sentences and calculate similarities

	corpus := make([]embedding, 0, 10000)
	data, _ := loadSICK()
	for i, v := range data {
		text := strings.TrimSpace(v.Pair[0])
		vector, _ := llm.EmbedText(text)
		g.Add(search.MakeNode(i, vector))
		b.Add(vector, text)
		corpus = append(corpus, embedding{
			Vector: vector,
			Text:   text,
		})
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
			embedding, _ := llm.EmbedText(query)

			start := time.Now()
			results := g.Search(embedding, 50)

			// Sort the results by similarity to the query
			sort.Slice(results, func(i, j int) bool {
				return search.Cosine(results[i].Value, embedding) > search.Cosine(results[j].Value, embedding)
			})

			// Print the results
			fmt.Printf("found %d results (%v)\n", len(results), time.Since(start))
			for i, v := range results {
				printResult(embedding, v.Value, corpus[v.Key].Text)
				if i >= 10 {
					break
				}
			}

			// Print the truth
			truth := b.Search(embedding, 50)
			fmt.Println("exact results:", len(truth))
			for i, v := range truth {
				printResult(embedding, v.Vector, v.Value)
				if i >= 10 {
					break
				}
			}
		}
	}
}

func mainSearchLSH() {

	// Load your language model
	llm := loadModel()
	defer llm.Close()

	lsh := search.NewLSH[*embedding](10000, 384)
	b := search.NewBag[string](384)

	// Embed the sentences and calculate similarities
	data, _ := loadSICK()
	uniq := make(map[string]struct{})
	for _, v := range data {
		text := strings.TrimSpace(v.Pair[0])
		if _, ok := uniq[text]; ok {
			continue
		}

		uniq[text] = struct{}{}
		vector, _ := llm.EmbedText(text)
		b.Add(vector, text)
		lsh.Add(vector, &embedding{
			Vector: vector,
			Text:   text,
		})
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
			embedding, _ := llm.EmbedText(query)

			start := time.Now()
			results := lsh.Query(embedding)

			// Sort the results by similarity to the query
			sort.Slice(results, func(i, j int) bool {
				return search.Cosine(results[i].Vector, embedding) > search.Cosine(results[j].Vector, embedding)
			})

			// Print the results
			fmt.Printf("found %d results (%v)\n", len(results), time.Since(start))
			for i, v := range results {
				printResult(embedding, v.Vector, v.Text)
				if i >= 10 {
					break
				}
			}

			// Print the truth
			truth := b.Search(embedding, 50)
			fmt.Println("exact results:", len(truth))
			for i, v := range truth {
				printResult(embedding, v.Vector, v.Value)
				if i >= 10 {
					break
				}
			}
		}
	}
}

func printResult(embedding, v []float32, text string) {
	dx := search.Cosine(embedding, v)
	switch {
	case dx >= 0.85:
		fmt.Printf(" ✅ %s (%.0f%%)\n", text, math.Round(dx*100))
	case dx >= 0.5:
		fmt.Printf(" ❔ %s (%.0f%%)\n", text, math.Round(dx*100))
	default:
		fmt.Printf(" ❌ %s (%.0f%%)\n", text, math.Round(dx*100))
	}
}

type entry struct {
	Pair  [2]string
	Rank  float64
	Label string
}

// loadSICK parses the SICK CSV dataset and returns sentence pairs with their relatedness scores
func loadSICK() ([]entry, error) {
	file, err := os.Open("dataset.txt")
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

		out = append(out, entry{
			Pair:  [2]string{sentenceA, sentenceB},
			Rank:  relatednessScore,
			Label: record[4],
		})
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

// mse calculates the mean squared error between two sets of scores
func mse(humanScores, predictedScores []float64) float64 {
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
