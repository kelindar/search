//go:build noasm || !(amd64 || (darwin && arm64) || (!darwin && arm64))

package simd

import "unsafe"

func f32_cosine_distance(x unsafe.Pointer, y unsafe.Pointer, result unsafe.Pointer, size uint64) {
	panic("not implemented")
}

func f32_dot_product(x unsafe.Pointer, y unsafe.Pointer, result unsafe.Pointer, size uint64) {
	panic("not implemented")
}
