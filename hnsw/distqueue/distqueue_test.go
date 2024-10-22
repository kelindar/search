package distqueue

import (
	"math/rand"
	"testing"
)

func TestQueue(t *testing.T) {

	pq := &DistQueueClosestFirst{}

	for i := 0; i < 10; i++ {
		pq.Push(rand.Uint32(), float32(rand.Float64()))
	}

	t.Log("Closest first, pop")
	ID, D := pq.Top()
	t.Logf("TOP before first top: %v %v", ID, D)
	var l float32 = 0.0
	for pq.Len() > 0 {
		item := pq.Pop()
		if item.D < l {
			t.Error("Incorrect order")
		}
		l = item.D
		t.Logf("%+v", item)
	}

	pq2 := &DistQueueClosestLast{}
	l = 1.0
	pq2.Init()
	pq2.Reserve(200) // try reserve
	for i := 0; i < 10; i++ {
		pq2.Push(rand.Uint32(), float32(rand.Float64()))
	}
	t.Log("Closest last, pop")
	for !pq2.Empty() {
		item := pq2.Pop()
		if item.D > l {
			t.Error("Incorrect order")
		}
		l = item.D
		t.Logf("%+v", item)
	}
}

func TestKBest(t *testing.T) {

	pq := &DistQueueClosestFirst{}
	pq.Reserve(5) // reserve less than needed
	for i := 0; i < 20; i++ {
		pq.Push(rand.Uint32(), rand.Float32())
	}

	// return K best matches, ordered as best first
	t.Log("closest last, still return K best")
	K := 10
	for pq.Len() > K {
		pq.Pop()
	}
	res := make([]*Item, K)
	for i := K - 1; i >= 0; i-- {
		res[i] = pq.Pop()
	}
	for i := 0; i < len(res); i++ {
		t.Logf("%+v", res[i])
	}
}
