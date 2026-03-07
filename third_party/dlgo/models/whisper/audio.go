package whisper

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// LoadWAV reads a WAV file and returns mono float32 samples at 16kHz.
// Supports 16-bit PCM WAV files. Resamples to 16kHz if needed.
func LoadWAV(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open WAV: %w", err)
	}
	defer f.Close()

	var riffHeader [12]byte
	if _, err := io.ReadFull(f, riffHeader[:]); err != nil {
		return nil, fmt.Errorf("read RIFF header: %w", err)
	}
	if string(riffHeader[:4]) != "RIFF" || string(riffHeader[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a valid WAV file")
	}

	var sampleRate uint32
	var numChannels uint16
	var bitsPerSample uint16
	var audioData []byte

	for {
		var chunkID [4]byte
		var chunkSize uint32
		if err := binary.Read(f, binary.LittleEndian, &chunkID); err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("read chunk ID: %w", err)
		}
		if err := binary.Read(f, binary.LittleEndian, &chunkSize); err != nil {
			return nil, fmt.Errorf("read chunk size: %w", err)
		}

		switch string(chunkID[:]) {
		case "fmt ":
			var audioFormat uint16
			binary.Read(f, binary.LittleEndian, &audioFormat)
			binary.Read(f, binary.LittleEndian, &numChannels)
			binary.Read(f, binary.LittleEndian, &sampleRate)
			f.Seek(4, io.SeekCurrent) // byte rate
			f.Seek(2, io.SeekCurrent) // block align
			binary.Read(f, binary.LittleEndian, &bitsPerSample)
			remaining := int64(chunkSize) - 16
			if remaining > 0 {
				f.Seek(remaining, io.SeekCurrent)
			}
		case "data":
			audioData = make([]byte, chunkSize)
			if _, err := io.ReadFull(f, audioData); err != nil {
				return nil, fmt.Errorf("read audio data: %w", err)
			}
		default:
			f.Seek(int64(chunkSize), io.SeekCurrent)
		}
	}

	if audioData == nil {
		return nil, fmt.Errorf("no audio data found in WAV")
	}
	if bitsPerSample != 16 {
		return nil, fmt.Errorf("only 16-bit PCM WAV supported, got %d-bit", bitsPerSample)
	}

	bytesPerSample := int(bitsPerSample) / 8
	totalSamples := len(audioData) / bytesPerSample
	samplesPerChannel := totalSamples / int(numChannels)

	raw := make([]float32, samplesPerChannel)
	for i := 0; i < samplesPerChannel; i++ {
		var sum float64
		for ch := 0; ch < int(numChannels); ch++ {
			offset := (i*int(numChannels) + ch) * bytesPerSample
			if offset+1 < len(audioData) {
				sample := int16(binary.LittleEndian.Uint16(audioData[offset:]))
				sum += float64(sample) / 32768.0
			}
		}
		raw[i] = float32(sum / float64(numChannels))
	}

	if sampleRate == 16000 {
		return raw, nil
	}

	return resample(raw, int(sampleRate), 16000), nil
}

func resample(samples []float32, fromRate, toRate int) []float32 {
	ratio := float64(toRate) / float64(fromRate)
	outLen := int(math.Ceil(float64(len(samples)) * ratio))
	out := make([]float32, outLen)

	for i := 0; i < outLen; i++ {
		srcPos := float64(i) / ratio
		srcIdx := int(srcPos)
		frac := float32(srcPos - float64(srcIdx))

		if srcIdx+1 < len(samples) {
			out[i] = samples[srcIdx]*(1-frac) + samples[srcIdx+1]*frac
		} else if srcIdx < len(samples) {
			out[i] = samples[srcIdx]
		}
	}
	return out
}

