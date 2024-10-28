// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package search

import (
	"compress/flate"
	"fmt"
	"io"
	"os"

	"github.com/kelindar/iostream"
)

// WriteTo writes the index to a writer.
func (b *Index[T]) WriteTo(dst io.Writer) (int64, error) {
	w := iostream.NewWriter(dst)
	i := w.Offset()

	// Write version
	if err := w.WriteUint8(1); err != nil {
		return 0, err
	}

	// Write the index
	err := w.WriteRange(len(b.arr), func(i int, w *iostream.Writer) error {
		if err := w.WriteFloat32s(b.arr[i].Vector); err != nil {
			return err
		}

		// Write the value (optional)
		switch v := any(b.arr[i].Value).(type) {
		case string:
			return w.WriteString(v)
		case []byte:
			return w.WriteBytes(v)
		default:
			return nil
		}
	})

	return w.Offset() - i, err
}

// ReadFrom reads the index from a reader.
func (b *Index[T]) ReadFrom(src io.Reader) (int64, error) {
	r := iostream.NewReader(src)
	s := r.Offset()

	// Read version
	version, err := r.ReadUint8()
	if err != nil {
		return 0, err
	}

	if version != 1 {
		return 0, fmt.Errorf("unsupported version: %d", version)
	}

	var length uint64
	if length, err = r.ReadUvarint(); err != nil {
		return r.Offset() - s, err
	}

	// Allocate space for the entries
	b.arr = make([]entry[T], length)
	for i := 0; i < int(length); i++ {

		// Read the vector
		if b.arr[i].Vector, err = r.ReadFloat32s(); err != nil {
			return r.Offset() - s, err
		}

		// Read the value (optional)
		switch any(b.arr[i].Value).(type) {
		case string:
			v, err := r.ReadString()
			if err != nil {
				return r.Offset() - s, err
			}
			b.arr[i].Value = any(v).(T)

		case []byte:
			v, err := r.ReadBytes()
			if err != nil {
				return r.Offset() - s, err
			}
			b.arr[i].Value = any(v).(T)
		}
	}

	return r.Offset() - s, nil
}

// ---------------------------------- File ----------------------------------

// WriteFile writes the index into a flate-compressed binary file.
func (idx *Index[T]) WriteFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}

	defer file.Close()
	writer, err := flate.NewWriter(file, flate.DefaultCompression)
	if err != nil {
		return err
	}

	// WriteTo the underlying writer
	defer writer.Close()
	_, err = idx.WriteTo(writer)
	return err
}

// ReadFile reads the index from a flate-compressed binary file.
func (idx *Index[T]) ReadFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}

	defer file.Close()
	_, err = idx.ReadFrom(flate.NewReader(file))
	return err
}
