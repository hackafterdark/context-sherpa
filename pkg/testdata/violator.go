package testdata

import "fmt"

func DoBadLogging() {
	fmt.Println("This is a direct print violation that the Sherpa should catch.")
}
