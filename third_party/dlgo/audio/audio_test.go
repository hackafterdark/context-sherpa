package audio

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func createTestWAV(t *testing.T, sampleRate uint32, bitsPerSample uint16, numChannels uint16, numSamples int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.wav")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	bytesPerSample := int(bitsPerSample) / 8
	dataSize := uint32(numSamples * int(numChannels) * bytesPerSample)

	// RIFF header
	f.Write([]byte("RIFF"))
	binary.Write(f, binary.LittleEndian, uint32(36+dataSize))
	f.Write([]byte("WAVE"))

	// fmt chunk
	f.Write([]byte("fmt "))
	binary.Write(f, binary.LittleEndian, uint32(16)) // chunk size
	if bitsPerSample == 32 {
		binary.Write(f, binary.LittleEndian, uint16(3)) // IEEE float
	} else {
		binary.Write(f, binary.LittleEndian, uint16(1)) // PCM
	}
	binary.Write(f, binary.LittleEndian, numChannels)
	binary.Write(f, binary.LittleEndian, sampleRate)
	byteRate := sampleRate * uint32(numChannels) * uint32(bytesPerSample)
	binary.Write(f, binary.LittleEndian, byteRate)
	blockAlign := numChannels * uint16(bytesPerSample)
	binary.Write(f, binary.LittleEndian, blockAlign)
	binary.Write(f, binary.LittleEndian, bitsPerSample)

	// data chunk
	f.Write([]byte("data"))
	binary.Write(f, binary.LittleEndian, dataSize)

	// Write samples
	for i := 0; i < numSamples*int(numChannels); i++ {
		val := math.Sin(float64(i) * 0.1 * math.Pi)
		switch bitsPerSample {
		case 16:
			sample := int16(val * 16384)
			binary.Write(f, binary.LittleEndian, sample)
		case 24:
			sample := int32(val * 4194304) // 2^22
			f.Write([]byte{byte(sample), byte(sample >> 8), byte(sample >> 16)})
		case 32:
			binary.Write(f, binary.LittleEndian, float32(val))
		}
	}

	return path
}

func TestLoadWAV16bit(t *testing.T) {
	path := createTestWAV(t, 16000, 16, 1, 1600) // 100ms at 16kHz
	samples, sr, err := LoadWAV(path)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}
	if sr != 16000 {
		t.Errorf("sample rate = %d, want 16000", sr)
	}
	if len(samples) != 1600 {
		t.Errorf("len(samples) = %d, want 1600", len(samples))
	}
	for _, s := range samples {
		if s < -1.0 || s > 1.0 {
			t.Errorf("sample out of range: %f", s)
			break
		}
	}
}

func TestLoadWAV24bit(t *testing.T) {
	path := createTestWAV(t, 16000, 24, 1, 800)
	samples, sr, err := LoadWAV(path)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}
	if sr != 16000 {
		t.Errorf("sample rate = %d, want 16000", sr)
	}
	if len(samples) != 800 {
		t.Errorf("len(samples) = %d, want 800", len(samples))
	}
}

func TestLoadWAV32bitFloat(t *testing.T) {
	path := createTestWAV(t, 16000, 32, 1, 800)
	samples, sr, err := LoadWAV(path)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}
	if sr != 16000 {
		t.Errorf("sample rate = %d, want 16000", sr)
	}
	if len(samples) != 800 {
		t.Errorf("len(samples) = %d, want 800", len(samples))
	}
}

func TestLoadWAVStereoToMono(t *testing.T) {
	path := createTestWAV(t, 16000, 16, 2, 800)
	samples, _, err := LoadWAV(path)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}
	if len(samples) != 800 {
		t.Errorf("len(samples) = %d, want 800 (stereo downmixed)", len(samples))
	}
}

func TestLoadWAVResample(t *testing.T) {
	path := createTestWAV(t, 44100, 16, 1, 4410) // 100ms at 44.1kHz
	samples, sr, err := LoadWAV(path)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}
	if sr != 16000 {
		t.Errorf("sample rate = %d, want 16000 (resampled)", sr)
	}
	expectedLen := int(float64(4410) / (44100.0 / 16000.0))
	if math.Abs(float64(len(samples)-expectedLen)) > 2 {
		t.Errorf("len(samples) = %d, want ~%d (resampled)", len(samples), expectedLen)
	}
}

func TestLoadWAVInvalidPath(t *testing.T) {
	_, _, err := LoadWAV("/nonexistent/file.wav")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestLoadWAVNotRIFF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.wav")
	os.WriteFile(path, []byte("NOT_RIFF_DATA_ABCD"), 0644)
	_, _, err := LoadWAV(path)
	if err == nil {
		t.Error("expected error for non-RIFF file")
	}
}

func TestResample(t *testing.T) {
	// Simple test: 100 samples at 48kHz → should produce ~33 at 16kHz
	input := make([]float32, 100)
	for i := range input {
		input[i] = float32(i)
	}

	output := resample(input, 48000, 16000)
	expectedLen := 33 // 100 samples at 48kHz → ~33 at 16kHz (ratio 3:1)
	if math.Abs(float64(len(output)-expectedLen)) > 2 {
		t.Errorf("resample len = %d, want ~%d", len(output), expectedLen)
	}

	// First sample should be close to 0
	if math.Abs(float64(output[0])) > 1.0 {
		t.Errorf("resample[0] = %f, want ~0", output[0])
	}
}

func TestApplySTFTOutputShape(t *testing.T) {
	chunk := make([]float32, 512)
	for i := range chunk {
		chunk[i] = float32(math.Sin(float64(i) * 0.05))
	}

	basis := make([]float32, 258*256)
	for i := range basis {
		basis[i] = 0.001
	}

	mag := applySTFT(chunk, basis)
	if len(mag) != 129 {
		t.Fatalf("freq bins = %d, want 129", len(mag))
	}
	for i, row := range mag {
		if len(row) != 4 {
			t.Fatalf("mag[%d] time steps = %d, want 4", i, len(row))
		}
	}
}

func TestLoadGeneratedWAV(t *testing.T) {
	path := filepath.Join("..", "testdata", "test_audio.wav")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("testdata not found")
	}

	samples, sr, err := LoadWAV(path)
	if err != nil {
		t.Fatalf("LoadWAV: %v", err)
	}
	if sr != 16000 {
		t.Errorf("sample rate = %d, want 16000", sr)
	}
	if len(samples) != 16000 {
		t.Errorf("len(samples) = %d, want 16000 (1 second)", len(samples))
	}

	// Verify it's a sine tone — samples should be in [-1, 1]
	for i, s := range samples {
		if s < -1.0 || s > 1.0 {
			t.Errorf("sample[%d] = %f out of [-1, 1]", i, s)
			break
		}
	}
}

func TestApplySTFTMagnitudeNonNegative(t *testing.T) {
	chunk := make([]float32, 512)
	for i := range chunk {
		chunk[i] = float32(math.Sin(float64(i)*0.3)) * 0.8
	}

	basis := make([]float32, 258*256)
	for i := range basis {
		basis[i] = float32(math.Sin(float64(i)*0.01)) * 0.01
	}

	mag := applySTFT(chunk, basis)
	for freq := range mag {
		for ts := range mag[freq] {
			if mag[freq][ts] < 0 {
				t.Errorf("negative magnitude at [%d][%d] = %f", freq, ts, mag[freq][ts])
			}
		}
	}
}
