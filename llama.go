package llm

/*
#cgo linux LDFLAGS: -ldl
#cgo darwin LDFLAGS: -ldl

#include <stdlib.h>
#include <llama.cpp>
*/
import "C"
import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"
)

// Model represents a loaded LLM/Embedding model.
type Model struct {
	handle C.llama_ctx // Handle to the model
}

// New creates a new  model from the given model file.
func New(modelPath string) (*Model, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.load_model(cPath)
	if handle == nil {
		return nil, fmt.Errorf("failed to load model: %s", C.GoString(C.get_error()))
	}

	return &Model{
		handle: handle,
	}, nil
}

// Close closes the model and releases any resources associated with it.
func (m *Model) Close() error {
	C.free_model(m.handle)
	return nil
}

// --------------------------------- Library Lookup ---------------------------------

// libpath is the path to the library.
var libpath string

func init() {
	var err error
	if libpath, err = findLlama(); err != nil {
		panic(err)
	}

	if err := load(libpath); err != nil {
		panic(err)
	}
}

// load loads the shared library at the given path.
func load(libPath string) error {
	cPath := C.CString(libPath)
	defer C.free(unsafe.Pointer(cPath))

	if ret := C.load_library(cPath); ret != 0 {
		return fmt.Errorf("failed to load %s: %s", libPath, C.GoString(C.get_error()))
	}
	return nil
}

// findLlama searches for the dynamic library in standard system paths.
func findLlama() (string, error) {
	switch runtime.GOOS {
	case "windows":
		return findLibrary("llama.dll", runtime.GOOS)
	case "darwin":
		return findLibrary("libllama.dylib", runtime.GOOS)
	default:
		return findLibrary("libllama.so", runtime.GOOS)
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
