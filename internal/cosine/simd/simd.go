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
func cosine(x, y []float32) float64 {
	var sum_xy, sum_xx, sum_yy float64
	for i := range x {
		sum_xy += float64(x[i] * y[i])
		sum_xx += float64(x[i] * x[i])
		sum_yy += float64(y[i] * y[i])
	}

	denominator := math.Sqrt(sum_xx) * math.Sqrt(sum_yy)
	if denominator == 0 {
		return 0.0
	}

	return sum_xy / denominator
}
