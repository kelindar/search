// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package search

import (
	"sort"

	"github.com/kelindar/search/internal/cosine/simd"
)

type Vector = []float32

type entry[T any] struct {
	Vector []float32
	Value  T
}

// Result represents a search result.
type Result[T any] struct {
	entry[T]
	Relevance float64 // The relevance of the result
}

// Index represents a brute-force search index, returning exact results.
type Index[T any] struct {
	arr []entry[T]
}

// NewIndex creates a new exact search index.
func NewIndex[T any]() *Index[T] {
	return &Index[T]{
		arr: make([]entry[T], 0),
	}
}

// Add adds a new vector to the search index.
func (b *Index[T]) Add(vx Vector, item T) {
	b.arr = append(b.arr, entry[T]{
		Vector: vx,
		Value:  item,
	})
}

// Search searches the index for the k-nearest neighbors of the query vector.
func (b *Index[T]) Search(query Vector, k int) []Result[T] {
	if k <= 0 {
		return nil
	}

	var relevance float64
	dst := make(minheap[T], 0, k)
	for _, v := range b.arr {
		simd.Cosine(&relevance, v.Vector, query)
		result := Result[T]{
			entry:     v,
			Relevance: relevance,
		}

		// If the heap is not full, add the result, otherwise replace
		// the minimum element
		switch {
		case dst.Len() < k:
			dst.Push(result)
		case result.Relevance > dst[0].Relevance:
			dst.Pop()
			dst.Push(result)
		}
	}

	// Sort the results by relevance
	sort.Sort(&dst)
	return dst
}

// --------------------------------- Heap ---------------------------------

// minheap is a min-heap of top values, ordered by relevance.
type minheap[T any] []Result[T]

// Len, Less, Swap implement the sort.Interface.
func (h *minheap[T]) Len() int           { return len(*h) }
func (h *minheap[T]) Less(i, j int) bool { return (*h)[i].Relevance > (*h)[j].Relevance }
func (h *minheap[T]) Swap(i, j int)      { (*h)[i], (*h)[j] = (*h)[j], (*h)[i] }

// Push adds a new element to the heap.
func (h *minheap[T]) Push(x Result[T]) {
	*h = append(*h, x)
	h.up(h.Len() - 1)
}

// Pop returns the minimum element from the heap.
func (h *minheap[T]) Pop() Result[T] {
	n := h.Len() - 1
	h.Swap(0, n)
	h.down(0, n)

	// Pop the last element
	x := (*h)[n]
	*h = (*h)[:n]
	return x
}

func (h minheap[T]) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || !(h[j].Relevance < h[i].Relevance) {
			break
		}

		h[i], h[j] = h[j], h[i]
		j = i
	}
}

func (h minheap[T]) down(at, n int) bool {
	i := at
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}
		j := j1 // left child
		if j2 := j1 + 1; j2 < n && (h[j2].Relevance < h[j1].Relevance) {
			j = j2 // = 2*i + 2  // right child
		}
		if h[i].Relevance < h[j].Relevance {
			break
		}

		h[i], h[j] = h[j], h[i]
		i = j
	}
	return i > at
}
