//go:build ignore

package main

import "os"

// a function that returns an error
func mightFail() error {
	return os.ErrNotExist
}

func main() {
	// This is a violation: the error is not checked.
	mightFail()
}
