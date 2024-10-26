package search

import (
	"encoding/gob"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
cpu: 13th Gen Intel(R) Core(TM) i7-13700K
BenchmarkBruteForce-24    	    4278	    286199 ns/op	     744 B/op	       5 allocs/op
*/
func BenchmarkBruteForce(b *testing.B) {
	data, err := loadDataset()
	assert.NoError(b, err)

	bag := NewExact[string]()
	for _, entry := range data {
		bag.Add(entry.Vector, entry.Label)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for v := range bag.Search(data[i%1000].Vector, 5) {
			_ = v
		}
	}
}

type record struct {
	Pair   [2]string `gob:"pair"`
	Rank   float64   `gob:"rank"`
	Label  string    `gob:"label"`
	Vector []float32 `gob:"vector"`
}

func loadDataset() ([]record, error) {
	file, err := os.Open("dataset.gob")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var data []record
	r := gob.NewDecoder(file)
	if err := r.Decode(&data); err != nil {
		return nil, err
	}

	return data, nil
}
