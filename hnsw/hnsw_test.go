package hnsw

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestHNSW_AddNode tests adding nodes to the HNSW index.
func TestHNSW_AddNode(t *testing.T) {
	tests := []struct {
		name          string
		initialNodes  int
		vectorsToAdd  [][]float32
		expectedCount int
	}{
		{
			name:          "Add single node",
			initialNodes:  0,
			vectorsToAdd:  [][]float32{{1, 0, 0}},
			expectedCount: 1,
		},
		{
			name:          "Add multiple nodes",
			initialNodes:  0,
			vectorsToAdd:  [][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			expectedCount: 3,
		},
		{
			name:          "Add duplicate nodes",
			initialNodes:  0,
			vectorsToAdd:  [][]float32{{1, 1, 1}, {1, 1, 1}},
			expectedCount: 2,
		},
		{
			name:          "Add nodes to existing index",
			initialNodes:  2,
			vectorsToAdd:  [][]float32{{1, 0, 0}, {0, 1, 0}},
			expectedCount: 4,
		},
	}

	for _, tt := range tests {
		tt := tt // capture range variable
		t.Run(tt.name, func(t *testing.T) {
			h := NewHNSW(5, 10)
			// Pre-populate nodes if initialNodes > 0
			for i := 0; i < tt.initialNodes; i++ {
				h.AddNode([]float32{float32(i), float32(i + 1), float32(i + 2)})
			}
			// Add new nodes
			for _, vec := range tt.vectorsToAdd {
				h.AddNode(vec)
			}
			assert.Equal(t, tt.expectedCount, len(h.nodes), "Number of nodes should be %d", tt.expectedCount)
		})
	}
}

// TestHNSW_BuildIndex tests building the HNSW index.
func TestHNSW_BuildIndex(t *testing.T) {
	tests := []struct {
		name          string
		vectors       [][]float32
		M             int
		expectedConns []struct{ Node, ConnectedTo int }
	}{
		{
			name:    "Empty index",
			vectors: [][]float32{},
			M:       5,
			// No connections expected
			expectedConns: []struct{ Node, ConnectedTo int }{},
		},
		{
			name: "Single node",
			vectors: [][]float32{
				{1, 0, 0},
			},
			M: 5,
			// No connections expected since there's only one node
			expectedConns: []struct{ Node, ConnectedTo int }{},
		},
		{
			name: "Two nodes",
			vectors: [][]float32{
				{1, 0, 0},
				{0, 1, 0},
			},
			M: 1,
			expectedConns: []struct{ Node, ConnectedTo int }{
				{0, 1},
				{1, 0},
			},
		},
		{
			name: "Multiple nodes",
			vectors: [][]float32{
				{1, 0, 0}, // 0
				{0, 1, 0}, // 1
				{0, 0, 1}, // 2
				{1, 1, 0}, // 3
				{1, 0, 1}, // 4
				{0, 1, 1}, // 5
				{1, 1, 1}, // 6
			},
			M: 2,
			// Exact connections depend on searchKNN implementation; we'll check that each node has <= M connections
			expectedConns: []struct{ Node, ConnectedTo int }{},
		},
	}

	for _, tt := range tests {
		tt := tt // capture range variable
		t.Run(tt.name, func(t *testing.T) {
			h := NewHNSW(tt.M, 10)
			for _, vec := range tt.vectors {
				h.AddNode(vec)
			}
			h.BuildIndex()

			if len(tt.expectedConns) > 0 {
				for _, conn := range tt.expectedConns {
					if conn.Node >= len(h.nodes) || conn.ConnectedTo >= len(h.nodes) {
						t.Errorf("Invalid node indices: %d connected to %d", conn.Node, conn.ConnectedTo)
						continue
					}
					assert.Contains(t, h.nodes[conn.Node].neighbors, conn.ConnectedTo, "Node %d should be connected to %d", conn.Node, conn.ConnectedTo)
				}
			} else {
				// For multiple nodes, ensure each node has at most M neighbors
				for i, node := range h.nodes {
					assert.LessOrEqual(t, len(node.neighbors), tt.M, "Node %d should have at most %d neighbors", i, tt.M)
				}
			}
		})
	}
}

// TestHNSW_Search tests the search functionality.
func TestHNSW_Search(t *testing.T) {
	tests := []struct {
		name          string
		vectors       [][]float32
		M             int
		query         []float32
		k             int
		expectedCount int
		expectError   bool
	}{
		{
			name:          "Empty index",
			vectors:       [][]float32{},
			M:             5,
			query:         []float32{1, 0, 0},
			k:             3,
			expectedCount: 0,
			expectError:   true,
		},
		{
			name: "Single node search",
			vectors: [][]float32{
				{1, 0, 0},
			},
			M:             5,
			query:         []float32{1, 0, 0},
			k:             1,
			expectedCount: 1,
			expectError:   false,
		},
		{
			name: "Search with k greater than nodes",
			vectors: [][]float32{
				{1, 0, 0},
				{0, 1, 0},
			},
			M:             1,
			query:         []float32{1, 1, 0},
			k:             5,
			expectedCount: 2,
			expectError:   false,
		},
		{
			name: "Exact match search",
			vectors: [][]float32{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			M:             2,
			query:         []float32{0, 1, 0},
			k:             1,
			expectedCount: 1,
			expectError:   false,
		},
		{
			name: "Search with multiple nearest neighbors",
			vectors: [][]float32{
				{1, 0, 0}, // 0
				{0, 1, 0}, // 1
				{0, 0, 1}, // 2
				{1, 1, 0}, // 3
				{1, 0, 1}, // 4
				{0, 1, 1}, // 5
				{1, 1, 1}, // 6
			},
			M:             3,
			query:         []float32{1, 0.5, 0.5},
			k:             3,
			expectedCount: 3,
			expectError:   false,
		},
		{
			name: "Search with zero vector query",
			vectors: [][]float32{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			M:           2,
			query:       []float32{0, 0, 0},
			k:           2,
			expectError: true,
		},
		{
			name: "Search with negative components",
			vectors: [][]float32{
				{1, -1, 0}, // 0
				{-1, 1, 0}, // 1
				{0, 0, 1},  // 2
			},
			M:             2,
			query:         []float32{1, -1, 0},
			k:             2,
			expectedCount: 2,
			expectError:   false,
		},
		{
			name: "High dimensional vectors",
			vectors: func() [][]float32 {
				vectors := make([][]float32, 100)
				for i := 0; i < 100; i++ {
					vec := make([]float32, 128)
					for j := 0; j < 128; j++ {
						vec[j] = float32(i + j)
					}
					vectors[i] = vec
				}
				return vectors
			}(),
			M: 10,
			query: func() []float32 {
				vec := make([]float32, 128)
				for j := 0; j < 128; j++ {
					vec[j] = float32(50 + j)
				}
				return vec
			}(),
			k:             5,
			expectedCount: 5,
			expectError:   false,
		},
		{
			name: "Duplicate vectors",
			vectors: [][]float32{
				{1, 1, 1}, // 0
				{1, 1, 1}, // 1
				{0, 0, 1}, // 2
			},
			M:             2,
			query:         []float32{1, 1, 1},
			k:             2,
			expectedCount: 2,
			expectError:   false,
		},
	}

	for _, tt := range tests {
		tt := tt // capture range variable
		t.Run(tt.name, func(t *testing.T) {
			h := NewHNSW(tt.M, 10)
			for _, vec := range tt.vectors {
				h.AddNode(vec)
			}
			h.BuildIndex()

			neighbors, err := h.Search(tt.query, tt.k)
			if tt.expectError {
				assert.Error(t, err, "Expected an error but got none")
			} else {
				assert.NoError(t, err, "Did not expect an error but got one")
				assert.Len(t, neighbors, tt.expectedCount, "Expected %d neighbors, got %d", tt.expectedCount, len(neighbors))
			}
		})
	}
}

// TestHNSW_SearchResults tests the actual correctness of search results.
func TestHNSW_SearchResults(t *testing.T) {
	h := NewHNSW(3, 10)
	vectors := [][]float32{
		{1, 0, 0}, // Index 0
		{0, 1, 0}, // Index 1
		{0, 0, 1}, // Index 2
		{1, 1, 0}, // Index 3
		{1, 0, 1}, // Index 4
		{0, 1, 1}, // Index 5
		{1, 1, 1}, // Index 6
	}
	for _, vec := range vectors {
		h.AddNode(vec)
	}
	h.BuildIndex()

	tests := []struct {
		name            string
		query           []float32
		k               int
		expectedIndices []int
	}{
		{
			name:            "Query close to index 0",
			query:           []float32{1, 0, 0},
			k:               1,
			expectedIndices: []int{0},
		},
		{
			name:            "Query close to index 6",
			query:           []float32{1, 1, 1},
			k:               3,
			expectedIndices: []int{6, 3, 5}, // Expected closest to (1,1,1)
		},
	}

	for _, tt := range tests {
		tt := tt // capture range variable
		t.Run(tt.name, func(t *testing.T) {
			neighbors, err := h.Search(tt.query, tt.k)
			assert.NoError(t, err, "Search should not return an error")
			assert.Len(t, neighbors, len(tt.expectedIndices), "Number of neighbors returned should be %d", len(tt.expectedIndices))
			// Check if expected indices are in the result
			for _, expected := range tt.expectedIndices {
				assert.Contains(t, neighbors, expected, "Expected neighbor %d to be in the search results", expected)
			}
		})
	}
}
