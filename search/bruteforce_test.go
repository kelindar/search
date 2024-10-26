package search

import (
	"encoding/gob"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
cpu: 13th Gen Intel(R) Core(TM) i7-13700K
BenchmarkBruteForce-24    	     889	   1311187 ns/op	  147752 B/op	       5 allocs/op
*/
func BenchmarkBruteForce(b *testing.B) {
	data, err := loadDataset()
	assert.NoError(b, err)

	bag := NewBag[string](384)
	for _, entry := range data {
		bag.Add(entry.Vector, entry.Label)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bag.Search(data[i%1000].Vector, 5)
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

/*func BenchmarkBruteForce(b *testing.B) {
	entries, err := loadSICK()
	if err != nil {
		b.Fatal(err)
	}

	llm := loadModel()
	defer llm.Close()

	var data []entry

	uniq := make(map[string]struct{})

	for _, entry := range entries {
		text1 := entry.Pair[0]
		if _, ok := uniq[text1]; !ok {
			uniq[entry.Label] = struct{}{}
			entry.Vector, _ = llm.EmbedText(entry.Pair[0])
			data = append(data, entry)
		}

		text2 := entry.Pair[1]
		if _, ok := uniq[text2]; !ok {
			uniq[entry.Label] = struct{}{}
			entry.Vector, _ = llm.EmbedText(entry.Pair[1])
			data = append(data, entry)
		}
	}

	f, _ := os.OpenFile("dataset.gob", os.O_CREATE|os.O_RDWR, 0644)
	w := gob.NewEncoder(f)
	w.Encode(data)
	f.Close()

}

func loadModel() *llm.Model {
	model := "../dist/MiniLM-L6-v2.Q8_0.gguf"
	fmt.Printf("Loading model: %s\n", model)

	mod, _ := filepath.Abs(model)
	ctx, err := llm.New(mod, 0)
	if err != nil {
		panic(err)
	}

	return ctx
}


// loadSICK parses the SICK CSV dataset and returns sentence pairs with their relatedness scores
func loadSICK() ([]entry, error) {
	file, err := os.Open("../eval/dataset.txt")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = '\t'    // Tab-separated file
	_, err = reader.Read() // Skip header line
	if err != nil {
		return nil, err
	}

	out := make([]entry, 0, 4600)
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}

		sentenceA := record[1]
		sentenceB := record[2]
		relatednessScore, err := strconv.ParseFloat(record[3], 64)
		if err != nil {
			return nil, err
		}

		out = append(out, entry{
			Pair:  [2]string{sentenceA, sentenceB},
			Rank:  relatednessScore,
			Label: record[4],
		})
	}

	return out, nil
}*/
