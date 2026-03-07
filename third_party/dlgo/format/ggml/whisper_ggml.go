package ggml

import (
	"encoding/binary"
	"fmt"
)

// LoadWhisperGGMLTensors loads a Whisper model from a GGML format file.
//
// This is a specialized wrapper around LoadGGMLModel() that handles Whisper-specific
// requirements. It returns the raw tensors and extracts model hyperparameters
// (metadata) from the file.
//
// Parameters:
//   - path: Path to the Whisper GGML model file
//
// Returns:
//   - tensors: Map of tensor name → RawTensor with loaded data
//   - metadata: Slice of 11 int32 hyperparameters:
//     [0] n_vocab: Vocabulary size
//     [1] n_audio_ctx: Audio context window size
//     [2] n_audio_state: Audio encoder hidden size
//     [3] n_audio_head: Number of audio attention heads
//     [4] n_audio_layer: Number of audio encoder layers
//     [5] n_text_ctx: Text context window size
//     [6] n_text_state: Text encoder hidden size
//     [7] n_text_head: Number of text attention heads
//     [8] n_text_layer: Number of text decoder layers
//     [9] n_mels: Number of mel spectrogram bands
//     [10] ftype: Model quantization type
//   - error: Any error encountered during loading
//
// Example usage:
//
//	tensors, metadata, err := ggml.LoadWhisperGGMLTensors("whisper-tiny.ggml")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Vocab size: %d\n", metadata[0])
//	fmt.Printf("Audio context: %d\n", metadata[1])
func LoadWhisperGGMLTensors(path string) (map[string]RawTensor, []int32, error) {
	metadataBytes, tensors, err := LoadGGMLModel(path)
	if err != nil {
		return nil, nil, err
	}

	// Extract metadata from the bytes
	metadata := make([]int32, len(metadataBytes)/4)
	for i := 0; i < len(metadata); i++ {
		metadata[i] = int32(binary.LittleEndian.Uint32(metadataBytes[i*4 : (i+1)*4]))
	}

	fmt.Printf("✓ Loaded %d tensors from GGML file\n", len(tensors))
	return tensors, metadata, nil
}