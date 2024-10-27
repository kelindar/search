package cosine

import "github.com/kelindar/search/internal/cosine/simd"

func CosineDistance(x, y []float32) float32 {
	if len(x) != len(y) {
		panic("vectors must have the same length")
	}

	return float32(simd.Cosine(x, y))
}
