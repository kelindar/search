// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package search

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLSH(t *testing.T) {
	lsh := NewLSH[string](10, 4)
	vectors := [][]float32{
		{100, 0, 0, 0},
		{0, 0, 0, 1},
	}

	for _, v := range vectors {
		lsh.Add(v, fmt.Sprintf("%+v", v))
	}

	neighbors := lsh.Query(vectors[0])
	assert.Equal(t, 1, len(neighbors))
	assert.Equal(t, "[100 0 0 0]", neighbors[0])
}

func TestEstimateLSH(t *testing.T) {
	testCases := []struct {
		name        string
		capacity    int
		dimensions  int
		probability float64
		hashes      int
		tables      int
		expectErr   bool
	}{
		{
			name:        "Typical case",
			capacity:    100000,
			dimensions:  384,
			probability: 0.05,
			hashes:      18,
			tables:      4,
		},
		{
			name:        "Minimum valid inputs",
			capacity:    1,
			dimensions:  1,
			probability: 0.5,
			hashes:      1, // Clamped to minimum
			tables:      1, // Clamped to minimum
		},
		{
			name:        "Maximum k clamp",
			capacity:    1000,
			dimensions:  1000000, // High dimensionality
			probability: 0.01,
			hashes:      64, // Clamped to maximum
			tables:      2,
		},
		{
			name:        "Invalid capacity",
			capacity:    -10,
			dimensions:  128,
			probability: 0.5,
			hashes:      0,
			tables:      0,
			expectErr:   true,
		},
		{
			name:        "Invalid p (zero)",
			capacity:    100,
			dimensions:  50,
			probability: 0.0,
			hashes:      0,
			tables:      0,
			expectErr:   true,
		},
		{
			name:        "Invalid p (greater than one)",
			capacity:    100,
			dimensions:  50,
			probability: 1.5,
			hashes:      0,
			tables:      0,
			expectErr:   true,
		},
	}

	for _, tc := range testCases {
		tc := tc // Capture range variable
		t.Run(tc.name, func(t *testing.T) {
			if tc.expectErr {
				assert.Panics(t, func() {
					estimateLSH(tc.capacity, tc.dimensions, tc.probability)
				})
			} else {
				nf, nt := estimateLSH(tc.capacity, tc.dimensions, tc.probability)
				assert.Equal(t, tc.hashes, nf, "Mismatch in number of hash functions")
				assert.Equal(t, tc.tables, nt, "Mismatch in number of hash tables")
			}
		})
	}
}
