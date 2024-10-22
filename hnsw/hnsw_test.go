package hnsw

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

var prefix = "siftsmall/siftsmall"
var dataSize = 10000
var efSearch = []int{1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300, 400}
var queries []Point
var truth [][]uint32

func TestMain(m *testing.M) {
	// LOAD QUERIES AND GROUNDTRUTH
	fmt.Printf("Loading query records\n")
	queries, truth = loadQueriesFromFvec(prefix)
	os.Exit(m.Run())
}
func TestSaveLoad(t *testing.T) {
	h := buildIndex()
	testSearch(h)

	fmt.Printf("Saving to index.dat\n")
	err := h.Save("index.dat")
	assert.Nil(t, err)

	fmt.Printf("Loading from index.dat\n")
	h2, timestamp, err := Load("index.dat")
	assert.Nil(t, err)

	fmt.Printf("Index loaded, time saved was %v", time.Unix(timestamp, 0))

	fmt.Printf(h2.Stats())
	testSearch(h2)
}

func TestSIFT(t *testing.T) {
	h := buildIndex()
	testSearch(h)
}

func buildIndex() *Hnsw {
	// BUILD INDEX
	var p Point
	p = make([]float32, 128)
	h := New(4, 200, p)
	h.DelaunayType = 1
	h.Grow(dataSize)

	buildStart := time.Now()
	fmt.Printf("Loading data and building index\n")
	points := make(chan job)
	go loadDataFromFvec(prefix, points)
	buildFromChan(h, points)
	buildStop := time.Since(buildStart)
	fmt.Printf("Index build in %v\n", buildStop)
	fmt.Printf(h.Stats())

	return h
}

func testSearch(h *Hnsw) {
	// SEARCH
	for _, ef := range efSearch {
		fmt.Printf("Now searching with ef=%v\n", ef)
		bestPrecision := 0.0
		bestTime := 999.0
		for i := 0; i < 10; i++ {
			start := time.Now()
			p := search(h, queries, truth, ef)
			stop := time.Since(start)
			bestPrecision = math.Max(bestPrecision, p)
			bestTime = math.Min(bestTime, stop.Seconds()/float64(len(queries)))
		}
		fmt.Printf("Best Precision 10-NN: %v\n", bestPrecision)
		fmt.Printf("Best time: %v s (%v queries / s)\n", bestTime, 1/bestTime)
	}
}

type job struct {
	p  Point
	id uint32
}

func buildFromChan(h *Hnsw, points chan job) {
	var wg sync.WaitGroup
	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func() {
			for {
				job, more := <-points
				if !more {
					wg.Done()
					return
				}
				h.Add(job.p, job.id)
			}
		}()
	}
	wg.Wait()
}

func search(h *Hnsw, queries []Point, truth [][]uint32, efSearch int) float64 {
	var p int32
	var wg sync.WaitGroup
	l := runtime.NumCPU()
	b := len(queries) / l

	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func(queries []Point, truth [][]uint32) {
			for j := range queries {
				results := h.Search(queries[j], efSearch, 10)
				// calc 10-NN precision
				for results.Len() > 10 {
					results.Pop()
				}
				for _, item := range results.Items() {
					for k := 0; k < 10; k++ {
						// !!! Our index numbers starts from 1
						if int32(truth[j][k]) == int32(item.ID)-1 {
							atomic.AddInt32(&p, 1)
						}
					}
				}
			}
			wg.Done()
		}(queries[i*b:i*b+b], truth[i*b:i*b+b])
	}
	wg.Wait()
	return (float64(p) / float64(10*b*l))
}

func readFloat32(f *os.File) (float32, error) {
	bs := make([]byte, 4)
	_, err := f.Read(bs)
	return float32(math.Float32frombits(binary.LittleEndian.Uint32(bs))), err
}

func readUint32(f *os.File) (uint32, error) {
	bs := make([]byte, 4)
	_, err := f.Read(bs)
	return binary.LittleEndian.Uint32(bs), err
}

func loadQueriesFromFvec(prefix string) (queries []Point, truth [][]uint32) {
	f2, err := os.Open(prefix + "_query.fvecs")
	if err != nil {
		panic("couldn't open query data file")
	}
	defer f2.Close()
	queries = make([]Point, 10000)
	qcount := 0
	for {
		d, err := readUint32(f2)
		if err != nil {
			break
		}
		if d != 128 {
			panic("Wrong dimension for this test...")
		}
		queries[qcount] = make([]float32, 128)
		for i := 0; i < int(d); i++ {
			queries[qcount][i], err = readFloat32(f2)
		}
		qcount++
	}
	queries = queries[0:qcount] // resize it
	fmt.Printf("Read %v query records\n", qcount)
	fmt.Printf("Loading groundtruth\n")
	// load query Vectors
	f3, err := os.Open(prefix + "_groundtruth.ivecs")
	if err != nil {
		panic("couldn't open groundtruth data file")
	}
	defer f3.Close()
	truth = make([][]uint32, 10000)
	tcount := 0
	for {
		d, err := readUint32(f3)
		if err != nil {
			break
		}
		if d != 100 {
			panic("Wrong dimension for this test...")
		}
		vec := make([]uint32, d)
		for i := 0; i < int(d); i++ {
			vec[i], err = readUint32(f3)
		}
		truth[tcount] = vec
		tcount++
	}
	fmt.Printf("Read %v truth records\n", tcount)

	if tcount != qcount {
		panic("Count mismatch queries <-> groundtruth")
	}

	return queries, truth
}

func loadDataFromFvec(prefix string, points chan job) {
	f, err := os.Open(prefix + "_base.fvecs")
	if err != nil {
		panic("couldn't open data file")
	}
	defer f.Close()
	count := 1
	for {
		d, err := readUint32(f)
		if err != nil {
			break
		}
		if d != 128 {
			panic("Wrong dimension for this test...")
		}
		var vec Point
		vec = make([]float32, 128)
		for i := 0; i < int(d); i++ {
			vec[i], err = readFloat32(f)
		}
		points <- job{p: vec, id: uint32(count)}
		count++
		if count%1000 == 0 {
			fmt.Printf("Read %v records\n", count)
		}
	}
	close(points)
}
