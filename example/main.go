package main

import "github.com/kelindar/llm"

func main() {
	//m, err := llm.New("../dist/Llama-3.2-1B-Instruct-Q6_K_L.gguf", 0)
	m, err := llm.New("../dist/MiniLM-L6-v2.Q4_K_M.gguf", 0)
	if err != nil {
		panic(err)
	}

	defer m.Close()

	embeddings, err := m.EmbedText("Hello, world!")
	if err != nil {
		panic(err)
	}

	println(embeddings)
}
