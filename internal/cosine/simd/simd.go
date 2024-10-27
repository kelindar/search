package simd

import (
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/klauspost/cpuid/v2"
)

var (
	avx2     = cpuid.CPU.Supports(cpuid.AVX2) && cpuid.CPU.Supports(cpuid.FMA3)
	apple    = runtime.GOARCH == "arm64" && runtime.GOOS == "darwin"
	neon     = runtime.GOARCH == "arm64" && cpuid.CPU.Supports(cpuid.SVE)
	hardware = avx2 || apple || neon
)

var pool = sync.Pool{
	New: func() any {
		var x float64
		return &x
	},
}

// Cosine calculates the cosine similarity between two vectors
func Cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("vectors must be of same length")
	}

	switch {
	case hardware:
		out := pool.Get().(*float64)
		f32_cosine_distance(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(out), uint64(len(a)))
		result := *out // copy out
		pool.Put(out)
		return result
	default:
		return cosine(a, b)
	}
}

// cosine calculates the cosine similarity between two vectors
func cosine(vec1, vec2 []float32) float64 {
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
