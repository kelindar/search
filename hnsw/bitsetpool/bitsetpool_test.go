package bitsetpool

import (
	"math/rand"
	"testing"
	"time"

	"github.com/kelindar/bitmap"
)

func TestBitset(t *testing.T) {

	start2 := time.Now()
	for j := 0; j < 100000; j++ {
		b2 := make(map[uint32]bool)
		for i := 0; i < 100; i++ {
			n := rand.Intn(1000000)
			b2[uint32(n)] = true
			m := rand.Intn(1000000)
			if b2[uint32(m)] == false {
			}
		}
	}
	stop2 := time.Since(start2)
	t.Logf("map done in %v", stop2.Seconds())

	start := time.Now()
	for j := 0; j < 100000; j++ {
		var b1 bitmap.Bitmap
		for i := 0; i < 100; i++ {
			n := rand.Intn(1000000)
			b1.Set(uint32(n))
			m := rand.Intn(1000000)
			b1.Contains(uint32(m))
		}
	}
	stop := time.Since(start)
	t.Logf("bitset done in %v", stop.Seconds())

	start3 := time.Now()
	pool := New()
	for j := 0; j < 100000; j++ {
		id, b := pool.Get()
		for i := 0; i < 100; i++ {
			n := rand.Intn(1000000)
			b.Set(uint32(n))
			m := rand.Intn(1000000)
			b.Contains(uint32(m))
		}
		pool.Free(id)
	}
	stop3 := time.Since(start3)
	t.Logf("bitset pool done in %v", stop3.Seconds())
	t.Logf("Performance boost %.2f%%", 100*(1-stop3.Seconds()/stop2.Seconds()))
}
