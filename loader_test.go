// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package search

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLibDirs_Windows(t *testing.T) {
	ext, libs := findLibDirs("windows")
	assert.Equal(t, ".dll", ext)
	assert.NotEmpty(t, libs)
}

func TestLibDirs_Darwin(t *testing.T) {
	ext, libs := findLibDirs("darwin")
	assert.Equal(t, ".dylib", ext)
	assert.NotEmpty(t, libs)
}

func TestLibDirs_Linux(t *testing.T) {
	ext, libs := findLibDirs("linux")
	assert.Equal(t, ".so", ext)
	assert.NotEmpty(t, libs)
}

func TestFindLibrary_Err(t *testing.T) {
	_, err := findLibrary("nonexistent", "linux")
	assert.Error(t, err)
}
