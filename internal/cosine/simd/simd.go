package simd

import (
	"math"
	"runtime"
	"unsafe"

	"github.com/klauspost/cpuid/v2"
)

var (
	avx2     = cpuid.CPU.Supports(cpuid.AVX2) && cpuid.CPU.Supports(cpuid.FMA3)
	apple    = runtime.GOARCH == "arm64" && runtime.GOOS == "darwin"
	neon     = runtime.GOARCH == "arm64" && cpuid.CPU.Supports(cpuid.SVE)
	hardware = avx2 || apple || neon
)

// Cosine calculates the cosine similarity between two vectors and stores the result in the destination
func Cosine(dst *float64, a, b []float32) {
	if len(a) != len(b) {
		panic("vectors must be of same length")
	}

	switch {
	case hardware:
		f32_cosine_distance(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(dst), uint64(len(a)))
	default:
		*dst = cosine(a, b)
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
