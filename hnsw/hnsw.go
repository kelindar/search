package hnsw

import (
	"container/heap"
	"errors"
	"math"
)

// ----------------- Cosine Distance Function -----------------

// CosineDistance computes the cosine distance between two normalized vectors.
// Returns 1.0 if vectors have different lengths or if either is a zero vector.
func CosineDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return 1.0
	}

	var dot float32
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
	}

	// Check if both vectors are normalized (norm ≈ 1)
	// If not, treat as zero vectors to maintain consistency
	var normA, normB float32
	for i := 0; i < len(a); i++ {
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if math.Abs(float64(normA-1.0)) > 1e-4 || math.Abs(float64(normB-1.0)) > 1e-4 {
		return 1.0 // At least one vector was not normalized
	}

	return 1.0 - dot
}

// ----------------- Priority Queue for Search -----------------

type Item struct {
	nodeIndex int
	distance  float32
}

type PriorityQueue []Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	// We want a max-heap, so reverse the comparison
	return pq[i].distance > pq[j].distance
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x interface{}) {
	item := x.(Item)
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// ----------------- HNSW Implementation -----------------

type Node struct {
	vector    []float32
	neighbors []int
	layer     int
}

type HNSW struct {
	nodes    []Node
	M        int
	ef       int
	maxLayer int
}

// NewHNSW creates a new HNSW index with given parameters.
func NewHNSW(M, ef int) *HNSW {
	return &HNSW{
		nodes:    make([]Node, 0),
		M:        M,
		ef:       ef,
		maxLayer: 1,
	}
}

// AddNode adds a node to the HNSW index.
// For simplicity, we assume an immutable index and build the graph after all nodes are added.
func (h *HNSW) AddNode(vector []float32) {
	node := Node{
		vector:    vector,
		neighbors: make([]int, 0, h.M),
		layer:     1, // Single layer for minimal implementation
	}
	h.nodes = append(h.nodes, node)
}

// BuildIndex constructs the HNSW graph.
// This function normalizes all vectors and establishes mutual connections based on cosine distance.
func (h *HNSW) BuildIndex() {
	for i := 0; i < len(h.nodes); i++ {
		// Normalize the vector
		if err := normalize(h.nodes[i].vector); err != nil {
			// If normalization fails (zero vector), leave the vector as-is
			// CosineDistance will handle zero vectors appropriately
		}

		// Find M nearest neighbors excluding the node itself
		neighbors := h.searchKNN(h.nodes[i].vector, h.M, i)
		h.nodes[i].neighbors = neighbors

		// Also connect this node to its neighbors
		for _, neighbor := range neighbors {
			if neighbor == i {
				continue // Avoid self-loop
			}
			if len(h.nodes[neighbor].neighbors) < h.M && !contains(h.nodes[neighbor].neighbors, i) {
				h.nodes[neighbor].neighbors = append(h.nodes[neighbor].neighbors, i)
			}
		}
	}
}

// Helper function to normalize a vector in-place.
// Returns an error if the vector is a zero vector.
func normalize(v []float32) error {
	var norm float32
	for _, val := range v {
		norm += val * val
	}
	if norm == 0 {
		return errors.New("cannot normalize zero vector")
	}
	invNorm := float32(1.0 / math.Sqrt(float64(norm)))
	for i := range v {
		v[i] *= invNorm
	}
	return nil
}

// Helper function to check if a slice contains an integer.
func contains(slice []int, item int) bool {
	for _, v := range slice {
		if v == item {
			return true
		}
	}
	return false
}

// searchKNN finds the k nearest neighbors using a linear search, excluding the specified node.
// If exclude is -1, no exclusion is performed (used during search operations).
func (h *HNSW) searchKNN(query []float32, k int, exclude int) []int {
	pq := &PriorityQueue{}
	heap.Init(pq)
	for i, node := range h.nodes {
		if i == exclude {
			continue
		}
		dist := CosineDistance(query, node.vector)
		if pq.Len() < k {
			heap.Push(pq, Item{nodeIndex: i, distance: dist})
		} else if dist < (*pq)[0].distance {
			heap.Pop(pq)
			heap.Push(pq, Item{nodeIndex: i, distance: dist})
		}
	}

	results := make([]int, pq.Len())
	for i := pq.Len() - 1; i >= 0; i-- {
		item := heap.Pop(pq).(Item)
		results[i] = item.nodeIndex
	}
	return results
}

// Search finds the k nearest neighbors for the given query vector.
// The query vector is expected to be normalized for accurate cosine distance calculations.
func (h *HNSW) Search(query []float32, k int) ([]int, error) {
	// Normalize the query vector
	if err := normalize(query); err != nil {
		return nil, errors.New("cannot normalize query vector")
	}

	if len(h.nodes) == 0 {
		return nil, errors.New("empty index")
	}
	return h.searchKNN(query, k, -1), nil // -1 indicates no exclusion
}
