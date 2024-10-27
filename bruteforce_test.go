package search

import (
	"encoding/gob"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
cpu: 13th Gen Intel(R) Core(TM) i7-13700K
BenchmarkIndex/search-24        	    4401	    265430 ns/op	     264 B/op	       2 allocs/op
*/
func BenchmarkIndex(b *testing.B) {
	data, err := loadDataset()
	assert.NoError(b, err)

	index := NewIndex[string]()
	for _, entry := range data {
		index.Add(entry.Vector, entry.Pair[0])
	}

	b.Run("search", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = index.Search(data[i%1000].Vector, 5)
		}
	})

}

func TestIndex(t *testing.T) {
	data, err := loadDataset()
	assert.NoError(t, err)

	bag := NewIndex[string]()
	for _, entry := range data {
		bag.Add(entry.Vector, entry.Pair[0])
	}

	for i := 0; i < 1000; i++ {
		record := data[i]

		results := bag.Search(record.Vector, 5)
		assert.Equal(t, 5, len(results))
		assert.InDelta(t, 1, results[0].Relevance, 1e-4)
	}
}

type record struct {
	Pair   [2]string `gob:"pair"`
	Rank   float64   `gob:"rank"`
	Label  string    `gob:"label"`
	Vector []float32 `gob:"vector"`
}

func loadDataset() ([]record, error) {
	file, err := os.Open("dist/dataset.gob")
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
