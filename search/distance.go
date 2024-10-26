package search

import (
	"github.com/viant/vec/search"
)

// Cosine computes the cosine similarity between two vectors. Higher values
// indicate more similar vectors.
func Cosine(a, b []float32) float64 {
	return 1 - float64(cosineDistance(a, b))
}

// CosineScaled computes the cosine distance between two vectors.
func cosineDistance(a, b []float32) float32 {
	return search.Float32s(a).CosineDistance(b)
}

// DistanceFunc is a function that computes the distance between two vectors.
type DistanceFunc func(a, b []float32) float32
