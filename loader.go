// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package search

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ebitengine/purego"
)

// libptr is a pointer to the loaded dynamic library.
var libptr uintptr
var load_library func(log_level int) uintptr
var load_model func(path_model string, n_gpu_layers uint32) uintptr
var load_context func(model uintptr, ctx_size uint32, embeddings bool) uintptr
var free_model func(model uintptr)
var free_context func(ctx uintptr)
var embed_size func(model uintptr) int32
var embed_text func(model uintptr, text string, out_embeddings []float32, out_tokens *uint32) int

func init() {
	libpath, err := findLlama()
	if err != nil {
		panic(err)
	}
	if libptr, err = load(libpath); err != nil {
		panic(err)
	}

	// Load the library functions
	purego.RegisterLibFunc(&load_library, libptr, "load_library")
	purego.RegisterLibFunc(&load_model, libptr, "load_model")
	purego.RegisterLibFunc(&load_context, libptr, "load_context")
	purego.RegisterLibFunc(&free_model, libptr, "free_model")
	purego.RegisterLibFunc(&free_context, libptr, "free_context")
	purego.RegisterLibFunc(&embed_size, libptr, "embed_size")
	purego.RegisterLibFunc(&embed_text, libptr, "embed_text")

	// Initialize the library (Log level WARN)
	load_library(2)
}

// --------------------------------- Library Lookup ---------------------------------

// findLlama searches for the dynamic library in standard system paths.
func findLlama() (string, error) {
	switch runtime.GOOS {
	case "windows":
		return findLibrary("llama_go.dll", runtime.GOOS)
	case "darwin":
		return findLibrary("libllama_go.dylib", runtime.GOOS)
	default:
		return findLibrary("libllama_go.so", runtime.GOOS)
	}
}

// findLibrary searches for a dynamic library by name across standard system paths.
// It returns the full path to the library if found, or an error listing all searched paths.
func findLibrary(libName, goos string, dirs ...string) (string, error) {
	libExt, commonPaths := findLibDirs(goos)
	dirs = append(dirs, commonPaths...)

	// Append the correct extension if missing
	if !strings.HasSuffix(libName, libExt) {
		libName += libExt
	}

	// Include current working directory
	if cwd, err := os.Getwd(); err == nil {
		dirs = append(dirs, cwd)
	}

	// Iterate through directories and search for the library
	searched := make([]string, 0, len(dirs))
	for _, dir := range dirs {
		filename := filepath.Join(dir, libName)
		searched = append(searched, filename)
		if fi, err := os.Stat(filename); err == nil && !fi.IsDir() {
			return filename, nil // Library found
		}
	}

	// Construct error message listing all searched paths
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Library '%s' not found, checked following paths:\n", libName))
	for _, path := range searched {
		sb.WriteString(fmt.Sprintf(" - %s\n", path))
	}

	return "", errors.New(sb.String())
}

// findLibDirs returns the library extension, relevant environment path, and common library directories based on the OS.
func findLibDirs(goos string) (string, []string) {
	switch goos {
	case "windows":
		systemRoot := os.Getenv("SystemRoot")
		return ".dll", append(
			filepath.SplitList(os.Getenv("PATH")),
			filepath.Join(systemRoot, "System32"),
			filepath.Join(systemRoot, "SysWOW64"),
		)
	case "darwin":
		return ".dylib", append(
			filepath.SplitList(os.Getenv("DYLD_LIBRARY_PATH")),
			"/usr/lib",
			"/usr/local/lib",
		)
	default: // Unix/Linux
		return ".so", append(
			filepath.SplitList(os.Getenv("LD_LIBRARY_PATH")),
			"/lib",
			"/usr/lib",
			"/usr/local/lib",
		)
	}
}
