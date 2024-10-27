package simd

import (
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
cpu: 13th Gen Intel(R) Core(TM) i7-13700K
BenchmarkSIMD/cos-std-24         	14839326	        81.02 ns/op	       0 B/op	       0 allocs/op
BenchmarkSIMD/cos-acc-24         	66064378	        18.21 ns/op	       0 B/op	       0 allocs/op
BenchmarkSIMD/dot-std-24         	14868597	        81.11 ns/op	       0 B/op	       0 allocs/op
BenchmarkSIMD/dot-acc-24         	125554860	         9.564 ns/op	       0 B/op	       0 allocs/op
*/
func BenchmarkSIMD(b *testing.B) {
	x := randVec()
	y := randVec()

	b.Run("cos-std", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			genericCosine(x, y)
		}
	})

	b.Run("cos-acc", func(b *testing.B) {
		var out float64
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			Cosine(&out, x, y)
		}
	})

	b.Run("dot-std", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			genericDotProduct(x, y)
		}
	})

	b.Run("dot-acc", func(b *testing.B) {
		var out float64
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			DotProduct(&out, x, y)
		}
	})
}

func TestCosine(t *testing.T) {
	for i := 0; i < 100; i++ {
		x := randVec()
		y := randVec()

		var actual float64
		Cosine(&actual, x, y)
		expect := genericCosine(x, y)
		assert.InDelta(t, expect, actual, 1e-4, "expected %v, got %v", genericCosine(x, y), actual)
	}
}

func TestDotProduct(t *testing.T) {
	for i := 0; i < 100; i++ {
		x := randVec()
		y := randVec()

		var actual float64
		DotProduct(&actual, x, y)
		expect := genericDotProduct(x, y)
		assert.InDelta(t, expect, actual, 1e-4, "expected %v, got %v", expect, actual)
	}
}

func randVec() []float32 {
	v := make([]float32, 384)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}
