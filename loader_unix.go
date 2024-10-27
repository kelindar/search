//go:build !windows
// +build !windows

package search

import "github.com/ebitengine/purego"

func load(name string) (uintptr, error) {
	return purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}
