#!/bin/bash

# requires gocc: go install github.com/kelindar/gocc/cmd/gocc@latest


gocc cosine_avx.c --arch avx2 -O3 -o simd --package simd
gocc cosine_neon.c --arch neon -O3 -o simd --package simd
gocc cosine_apple.c --arch apple -O3 -o simd --package simd