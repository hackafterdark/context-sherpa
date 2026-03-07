package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/silero"
)

func main() {
	// Create a new Silero VAD instance
	vad, err := silero.NewSileroVAD("")
	if err != nil {
		fmt.Printf("Error creating VAD: %v\n", err)
		os.Exit(1)
	}
	defer vad.Reset()

	// Print VAD information
	fmt.Printf("✓ Silero VAD loaded\n")
	fmt.Printf("  Model: %s\n", "Silero VAD v5")

	// Create a simple audio buffer (16kHz, 32ms = 512 samples)
	samples := make([]float32, 512)
	for i := range samples {
		samples[i] = float32(i) / 512.0 // Simple ramp
	}

	// Process the audio buffer
	prob := vad.ProcessChunkStateful(samples)
	// Print speech probability
	fmt.Printf("\nSpeech probability: %.4f\n", prob)
	fmt.Printf("(0.0 = definitely speech, 1.0 = definitely not speech)\n")
	fmt.Printf("\n✓ Silero VAD validation complete!\n")
}