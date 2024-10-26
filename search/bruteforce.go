package search

import (
	"cmp"
	"sort"
)

type entry[T cmp.Ordered] struct {
	Vector []float32
	Value  T
}

type Result[T cmp.Ordered] struct {
	entry[T]
	Relevance float64 // The relevance of the result
}

type Bag[T cmp.Ordered] struct {
	arr []entry[T]
	dim int
}

func NewBag[T cmp.Ordered](dim int) *Bag[T] {
	return &Bag[T]{
		arr: make([]entry[T], 0),
		dim: dim,
	}
}

func (b *Bag[T]) Add(vx Vector, data T) {
	b.arr = append(b.arr, entry[T]{
		Vector: vx,
		Value:  data,
	})
}

// Search implements a brute-force search algorithm to find the k-nearest neighbors
func (b *Bag[T]) Search(query Vector, k int) []Result[T] {

	// Compute the distance to each vector
	results := make([]Result[T], len(b.arr))
	for i, v := range b.arr {
		results[i] = Result[T]{
			Relevance: Cosine(v.Vector, query),
			entry:     v,
		}
	}

	// Sort the distances
	sort.Slice(results, func(i, j int) bool {
		return results[i].Relevance > results[j].Relevance
	})

	return results[:k]
}
