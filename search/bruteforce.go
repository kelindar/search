package search

import (
	"cmp"
	"sort"
	"sync"
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
	arr  []entry[T]
	dim  int
	pool sync.Pool
}

func NewBag[T cmp.Ordered](dim int) *Bag[T] {
	return &Bag[T]{
		arr: make([]entry[T], 0),
		dim: dim,
		/*pool: sync.Pool{
			New: func() any {
				top := &topk[T]{}
				heap.Init(top)
				return top
			},
		},*/
	}
}

func (b *Bag[T]) Add(vx Vector, data T) {
	b.arr = append(b.arr, entry[T]{
		Vector: vx,
		Value:  data,
	})
}

// Search implements a brute-force search algorithm to find the k-nearest neighbors
func (b *Bag[T]) Search(query Vector, k int) []Result[T] { //iter.Seq[Result[T]] {
	if k <= 0 {
		return nil
	}

	/*h := b.pool.Get().(*topk[T])
	h.Reset()
	defer b.pool.Put(h)*/

	//h := &minheap[T]{}
	var h minheap[T]

	for _, v := range b.arr {
		relevance := Cosine(v.Vector, query)
		result := Result[T]{
			entry:     v,
			Relevance: relevance,
		}

		if h.Len() < k {
			//heap.Push(h, result)
			h.Push(result)
		} else if relevance > h[0].Relevance {
			//heap.Pop(h)
			//heap.Push(h, result)
			h.Pop()
			h.Push(result)
		}
	}

	sort.Slice(h, func(i, j int) bool {
		return h[i].Relevance > h[j].Relevance
	})

	return h[:k]

	// Extract results from the heap and sort them in descending order of relevance
	/*topResults := make([]Result[T], h.Len())
	for i := len(topResults) - 1; i >= 0; i-- {
		//topResults[i] = heap.Pop(h).(Result[T])
		topResults[i] = h.Pop()
	}

	return topResults*/
	/*return func(yield func(Result[T]) bool) {
		for h.Len() > 0 {
			if !yield(heap.Pop(h).(Result[T])) {
				return
			}
		}
	}*/
}

// --------------------------------- Heap ---------------------------------
/*
// topk is a min-heap that keeps the top k results with the highest relevance.
type topk[T cmp.Ordered] []Result[T]

// Implement heap.Interface for topKHeap
func (h topk[T]) Len() int           { return len(h) }
func (h topk[T]) Less(i, j int) bool { return h[i].Relevance < h[j].Relevance } // Min-heap
func (h topk[T]) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

// Push adds an element to the heap.
func (h *topk[T]) Push(x interface{}) {
	*h = append(*h, x.(Result[T]))
}

// Pop removes and returns the smallest element from the heap.
func (h *topk[T]) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

func (h *topk[T]) Reset() {
	*h = (*h)[:0]
}*/

// minheap is a min-heap of top values, ordered by count.
type minheap[T cmp.Ordered] []Result[T]

// Reset resets the minheap to an empty state.
func (h *minheap[T]) Reset() {
	*h = (*h)[:0]
}

// Len, Less, Swap implement the sort.Interface.
func (h *minheap[T]) Len() int           { return len(*h) }
func (h *minheap[T]) Less(i, j int) bool { return (*h)[i].Relevance < (*h)[j].Relevance }
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

// Update updates the count of the element at index i.
func (h minheap[T]) Update(i int, relevance float64) {
	h[i].Relevance = relevance
	if !h.down(i, len(h)) {
		h.up(i)
	}
}

// Clone clones the minheap into dst.
func (h minheap[T]) Clone(dst *minheap[T]) {
	for _, e := range h {
		if e.Relevance > 0 {
			(*dst) = append(*dst, e)
		}
	}
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
