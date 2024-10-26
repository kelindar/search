package search

import (
	"iter"
	"sort"
)

type entry[T any] struct {
	Vector []float32
	Value  T
}

type Result[T any] struct {
	entry[T]
	Relevance float64 // The relevance of the result
}

type Bag[T any] struct {
	arr []entry[T]
	dim int
}

// NewExact creates a new exact search index.
func NewExact[T any]() *Bag[T] {
	return &Bag[T]{
		arr: make([]entry[T], 0),
	}
}

// Add adds a new vector to the search index.
func (b *Bag[T]) Add(vx Vector, item T) {
	b.arr = append(b.arr, entry[T]{
		Vector: vx,
		Value:  item,
	})
}

// Search searches the index for the k-nearest neighbors of the query vector.
func (b *Bag[T]) Search(query Vector, k int) iter.Seq2[float64, T] {
	return func(yield func(float64, T) bool) {
		for _, r := range b.search(query, k) {
			if !yield(r.Relevance, r.Value) {
				break
			}
		}
	}
}

// Search implements a brute-force search algorithm to find the k-nearest neighbors
func (b *Bag[T]) search(query Vector, k int) []Result[T] {
	if k <= 0 {
		return nil
	}

	var h minheap[T]
	for _, v := range b.arr {
		relevance := Cosine(v.Vector, query)
		result := Result[T]{
			entry:     v,
			Relevance: relevance,
		}

		// If the heap is not full, add the result, otherwise replace
		// the minimum element
		switch {
		case h.Len() < k:
			h.Push(result)
		case relevance > h[0].Relevance:
			h.Pop()
			h.Push(result)
		}
	}

	// Sort the results by relevance
	sort.Sort(&h)
	return h
}

// --------------------------------- Heap ---------------------------------

// minheap is a min-heap of top values, ordered by relevance.
type minheap[T any] []Result[T]

// Reset resets the minheap to an empty state.
func (h *minheap[T]) Reset() {
	*h = (*h)[:0]
}

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
