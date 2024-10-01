//go:build !windows
// +build !windows

package llm

import "github.com/ebitengine/purego"

func load(name string) (uintptr, error) {
	return purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}
