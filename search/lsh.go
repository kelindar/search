package search

import (
	"math"
	"math/rand/v2"
)

// LSH implements Locality-Sensitive Hashing for cosine similarity
type LSH[T comparable] struct {
	nt, nf int              // Number of tables and hash functions
	planes [][][]float32    // Hyperplanes for each hash function in each table
	tables []map[uint64][]T // Hash tables mapping hash values to vector indices
	dim    int              // Dimensionality of input vectors
}

// NewLSH initializes a new LSH instance
func NewLSH[T comparable](capacity, dim int) *LSH[T] {
	nf, nt := estimateLSH(capacity, dim, 0.10) // 10% collision probability
	planes := make([][][]float32, nt)
	tables := make([]map[uint64][]T, nt)

	// Initialize hyperplanes for this table
	for i := 0; i < nt; i++ {
		planes[i] = make([][]float32, nf)
		for j := 0; j < nf; j++ {
			planes[i][j] = randPlane(dim)
		}

		// Initialize the hash table.
		tables[i] = make(map[uint64][]T)
	}

	return &LSH[T]{
		nt:     nt,
		nf:     nf,
		planes: planes,
		tables: tables,
		dim:    dim,
	}
}

// Add inserts a vector into the LSH tables.
// vector: The vector to insert.
// index: An identifier for the vector (e.g., its index in the dataset).
func (lsh *LSH[T]) Add(vector []float32, index T) {
	for i := 0; i < lsh.nt; i++ {
		hash := lsh.hash(vector, i)
		lsh.tables[i][hash] = append(lsh.tables[i][hash], index)
	}
}

// Query finds candidate neighbors for the given vector.
// Returns a slice of indices of candidate vectors.
func (lsh *LSH[T]) Query(vector []float32) []T {
	candidates := make(map[T]struct{}, 64)

	for i := 0; i < lsh.nt; i++ {
		hash := lsh.hash(vector, i)
		for _, idx := range lsh.tables[i][hash] {
			candidates[idx] = struct{}{}
		}
	}

	result := make([]T, 0, len(candidates))
	for idx := range candidates {
		result = append(result, idx)
	}

	return result
}

// hash computes the hash of a vector for a given table
func (lsh *LSH[T]) hash(vector []float32, table int) uint64 {
	var hash uint64

	for i, hp := range lsh.planes[table] {
		var dot float32
		for d := 0; d < lsh.dim; d++ {
			dot += vector[d] * hp[d]
		}
		if dot >= 0 {
			hash |= 1 << uint(i)
		}
	}

	return hash
}

// randPlane generates a new random hyperplane for a hash function
func randPlane(n int) []float32 {
	out := make([]float32, n)
	for d := 0; d < n; d++ {
		out[d] = float32(rand.Float32())
	}
	return out
}

// estimateLSH estimates the number of hash functions (k) and number of hash tables (L)
// based on the expected dataset size (capacity), vector dimensionality (dim), and desired
// collision probability (p).
func estimateLSH(capacity, dim int, p float64) (nf, nt int) {
	switch {
	case capacity <= 0:
		panic("capacity must be positive")
	case dim <= 0:
		panic("dimensionality must be positive")
	case p <= 0 || p >= 1:
		panic("invalid collision probability: require 0 < p < 1")
	}

	// Calculate the number of hash functions (k)
	// Incorporate dimensionality to ensure higher dimensions have more hash functions
	nf = int(math.Ceil(math.Log(1/p) * math.Log(float64(dim))))
	nf = max(min(nf, 64), 1) // Ensure 1 <= k <= 64

	// Calculate the number of hash tables (L)
	nt = int(math.Ceil(math.Log(float64(capacity)) / math.Log(1/p)))
	nt = max(nt, 1) // Ensure at least one hash table

	return nf, nt
}
