package main

import (
	"fmt"
	"os"
	"time"

	"github.com/computerex/dlgo/models/whisper"
)

func main() {
	modelPath := `C:\projects\evoke\models\whisper-base-q8_0.gguf`
	tokenizerPath := `C:\projects\evoke\models\whisper-base-onnx\tokenizer.json`
	audioPath := `C:\projects\evoke\jfk.wav`

	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}
	if len(os.Args) > 2 {
		tokenizerPath = os.Args[2]
	}
	if len(os.Args) > 3 {
		audioPath = os.Args[3]
	}

	fmt.Printf("Loading Whisper model from %s...\n", modelPath)
	start := time.Now()
	m, err := whisper.LoadWhisperModel(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Model loaded in %v\n", time.Since(start))
	fmt.Printf("  Encoder: %d layers, %d dim, %d heads\n",
		m.Config.NEncLayers, m.Config.DModel, m.Config.NHeads)
	fmt.Printf("  Decoder: %d layers, vocab %d\n",
		m.Config.NDecLayers, m.Config.NVocab)

	tok, err := whisper.LoadTokenizer(tokenizerPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading tokenizer: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nTranscribing %s...\n", audioPath)
	start = time.Now()
	text, err := m.TranscribeFile(audioPath, tok)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error transcribing: %v\n", err)
		os.Exit(1)
	}
	elapsed := time.Since(start)

	fmt.Printf("\nTranscription (%v):\n  %s\n", elapsed, text)
}
