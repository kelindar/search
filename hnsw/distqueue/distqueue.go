package distqueue

type Item struct {
	ID uint32
	D  float32
}

type DistQueueClosestFirst struct {
	initiated bool
	items     []*Item
	Size      int
}

func (pq *DistQueueClosestFirst) Init() *DistQueueClosestFirst {
	pq.items = make([]*Item, 1, pq.Size+1)
	pq.items[0] = nil // Heap queue first element should always be nil
	pq.initiated = true
	return pq
}

func (pq *DistQueueClosestFirst) Reset() {
	pq.items = pq.items[0:1]
}
func (pq *DistQueueClosestFirst) Items() []*Item {
	return pq.items[1:]
}
func (pq *DistQueueClosestFirst) Reserve(n int) {
	if n > len(pq.items)-1 {
		// reserve memory by setting the slice capacity
		items2 := make([]*Item, len(pq.items), n+1)
		copy(pq.items, items2)
		pq.items = items2
	}
}

// Push the value item into the priority queue with provided priority.
func (pq *DistQueueClosestFirst) Push(id uint32, d float32) *Item {
	if !pq.initiated {
		pq.Init()
	}
	item := &Item{ID: id, D: d}
	pq.items = append(pq.items, item)
	pq.swim(len(pq.items) - 1)
	return item
}

func (pq *DistQueueClosestFirst) PushItem(item *Item) {
	if !pq.initiated {
		pq.Init()
	}
	pq.items = append(pq.items, item)
	pq.swim(len(pq.items) - 1)
}

func (pq *DistQueueClosestFirst) Pop() *Item {
	if len(pq.items) <= 1 {
		return nil
	}
	var max = pq.items[1]
	//pq.items[1], pq.items[len(pq.items)-1] = pq.items[len(pq.items)-1], pq.items[1]
	pq.items[1], pq.items[len(pq.items)-1] = pq.items[len(pq.items)-1], pq.items[1]
	pq.items = pq.items[0 : len(pq.items)-1]
	pq.sink(1)
	return max
}

func (pq *DistQueueClosestFirst) Top() (uint32, float32) {
	if len(pq.items) <= 1 {
		return 0, 0
	}
	return pq.items[1].ID, pq.items[1].D
}

func (pq *DistQueueClosestFirst) Head() (uint32, float32) {
	if len(pq.items) <= 1 {
		return 0, 0
	}
	return pq.items[1].ID, pq.items[1].D
}

func (pq *DistQueueClosestFirst) Len() int {
	return len(pq.items) - 1
}

func (pq *DistQueueClosestFirst) Empty() bool {
	return len(pq.items) == 1
}

func (pq *DistQueueClosestFirst) swim(k int) {
	for k > 1 && (pq.items[k/2].D > pq.items[k].D) {
		pq.items[k], pq.items[k/2] = pq.items[k/2], pq.items[k]
		k = k / 2
	}
}

func (pq *DistQueueClosestFirst) sink(k int) {
	for 2*k <= len(pq.items)-1 {
		var j = 2 * k
		if j < len(pq.items)-1 && (pq.items[j].D > pq.items[j+1].D) {
			j++
		}
		if !(pq.items[k].D > pq.items[j].D) {
			break
		}
		pq.items[k], pq.items[j] = pq.items[j], pq.items[k]
		k = j
	}
}

type DistQueueClosestLast struct {
	initiated bool
	items     []*Item
	Size      int
}

func (pq *DistQueueClosestLast) Init() *DistQueueClosestLast {
	pq.items = make([]*Item, 1, pq.Size+1)
	pq.items[0] = nil // Heap queue first element should always be nil
	pq.initiated = true
	return pq
}

func (pq *DistQueueClosestLast) Items() []*Item {
	return pq.items[1:]
}
func (pq *DistQueueClosestLast) Reserve(n int) {
	if n > len(pq.items)-1 {
		// reserve memory by setting the slice capacity
		items2 := make([]*Item, len(pq.items), n+1)
		copy(pq.items, items2)
		pq.items = items2
	}
}

// Push the value item into the priority queue with provided priority.
func (pq *DistQueueClosestLast) Push(id uint32, d float32) *Item {
	if !pq.initiated {
		pq.Init()
	}
	item := &Item{ID: id, D: d}
	pq.items = append(pq.items, item)
	pq.swim(len(pq.items) - 1)
	return item
}

// PopAndPush pops the top element and adds a new to the heap in one operation which is faster than two seperate calls to Pop and Push
func (pq *DistQueueClosestLast) PopAndPush(id uint32, d float32) *Item {
	if !pq.initiated {
		pq.Init()
	}
	item := &Item{ID: id, D: d}
	pq.items[1] = item
	pq.sink(1)
	return item
}

func (pq *DistQueueClosestLast) PushItem(item *Item) {
	if !pq.initiated {
		pq.Init()
	}
	pq.items = append(pq.items, item)
	pq.swim(len(pq.items) - 1)
}

func (pq *DistQueueClosestLast) Pop() *Item {
	if len(pq.items) <= 1 {
		return nil
	}
	var max = pq.items[1]
	pq.items[1], pq.items[len(pq.items)-1] = pq.items[len(pq.items)-1], pq.items[1]
	pq.items = pq.items[0 : len(pq.items)-1]
	pq.sink(1)
	return max
}

func (pq *DistQueueClosestLast) Top() (uint32, float32) {
	if len(pq.items) <= 1 {
		return 0, 0
	}
	return pq.items[1].ID, pq.items[1].D
}

func (pq *DistQueueClosestLast) Head() (uint32, float32) {
	if len(pq.items) <= 1 {
		return 0, 0
	}
	return pq.items[1].ID, pq.items[1].D
}

func (pq *DistQueueClosestLast) Len() int {
	return len(pq.items) - 1
}

func (pq *DistQueueClosestLast) Empty() bool {
	return len(pq.items) == 1
}

func (pq *DistQueueClosestLast) swim(k int) {
	for k > 1 && (pq.items[k/2].D < pq.items[k].D) {
		pq.items[k], pq.items[k/2] = pq.items[k/2], pq.items[k]
		//pq.exch(k/2, k)
		k = k / 2
	}
}

func (pq *DistQueueClosestLast) sink(k int) {
	for 2*k <= len(pq.items)-1 {
		var j = 2 * k
		if j < len(pq.items)-1 && (pq.items[j].D < pq.items[j+1].D) {
			j++
		}
		if !(pq.items[k].D < pq.items[j].D) {
			break
		}
		pq.items[k], pq.items[j] = pq.items[j], pq.items[k]
		k = j
	}
}
