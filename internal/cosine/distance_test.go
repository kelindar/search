package cosine

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCosine(t *testing.T) {
	for i := 0; i < 100; i++ {
		x := randVec()
		y := randVec()

		expect := cosine(x, y)
		actual := CosineDistance(x, y)
		assert.InDelta(t, expect, actual, 1e-4, "expected %v, got %v", expect, actual)
	}
}

// cosine computes the cosine similarity between two vectors. Higher values
// indicate more similar vectors.
func cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("vectors must be of equal length")
	}

	dp, an, bn := float64(0), float64(0), float64(0)
	for i := range a {
		dp += float64(a[i] * b[i])
		an += float64(a[i] * a[i])
		bn += float64(b[i] * b[i])
	}

	return dp / (math.Sqrt(an) * math.Sqrt(bn))
}

func randVec() []float32 {
	v := make([]float32, 384)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}
