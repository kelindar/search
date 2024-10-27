package search

import (
	dist "github.com/kelindar/search/internal/cosine"
)

type Vector = []float32

// Cosine computes the cosine similarity between two vectors. Higher values
// indicate more similar vectors.
func Cosine(a, b Vector) float64 {
	return 1 - float64(cosineDistance(a, b))
}

// CosineScaled computes the cosine distance between two vectors.
func cosineDistance(a, b Vector) float32 {
	//return search.Float32s(a).CosineDistance(b)
	return dist.CosineDistance(a, b)
}
