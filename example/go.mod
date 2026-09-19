module github.com/kelindar/search/example

go 1.27.0

toolchain go1.27.1

require (
	github.com/kelindar/search v0.0.0
	github.com/kelindar/search/llama v0.0.0
	github.com/stretchr/testify v1.12.1
)

require (
	github.com/ebitengine/purego v0.11.0 // indirect
	github.com/kelindar/iostream v1.4.0 // indirect
	github.com/klauspost/cpuid/v2 v2.4.0 // indirect
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.48.0 // indirect
)

replace github.com/kelindar/search => ..

replace github.com/kelindar/search/llama => ../llama
