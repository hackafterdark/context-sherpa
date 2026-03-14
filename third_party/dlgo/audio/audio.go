package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// LoadWAV reads a WAV file and returns mono float32 samples resampled to 16kHz.
func LoadWAV(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("open: %w", err)
	}
	defer f.Close()

	// RIFF header
	var riffID [4]byte
	f.Read(riffID[:])
	if string(riffID[:]) != "RIFF" {
		return nil, 0, fmt.Errorf("not a RIFF file")
	}
	var chunkSize uint32
	binary.Read(f, binary.LittleEndian, &chunkSize)
	var waveID [4]byte
	f.Read(waveID[:])
	if string(waveID[:]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a WAVE file")
	}

	var sampleRate uint32
	var bitsPerSample uint16
	var numChannels uint16
	var audioFormat uint16
	var dataFound bool

	// Read sub-chunks
	for {
		var subID [4]byte
		_, err := f.Read(subID[:])
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, 0, fmt.Errorf("read chunk: %w", err)
		}

		var subSize uint32
		binary.Read(f, binary.LittleEndian, &subSize)

		switch string(subID[:]) {
		case "fmt ":
			binary.Read(f, binary.LittleEndian, &audioFormat)
			binary.Read(f, binary.LittleEndian, &numChannels)
			binary.Read(f, binary.LittleEndian, &sampleRate)
			var byteRate uint32
			binary.Read(f, binary.LittleEndian, &byteRate)
			var blockAlign uint16
			binary.Read(f, binary.LittleEndian, &blockAlign)
			binary.Read(f, binary.LittleEndian, &bitsPerSample)
			// Skip extra fmt bytes
			if subSize > 16 {
				f.Seek(int64(subSize-16), io.SeekCurrent)
			}

		case "data":
			dataFound = true

			var samples []float32
			switch bitsPerSample {
			case 16:
				nSamples := subSize / 2
				raw := make([]int16, nSamples)
				if err := binary.Read(f, binary.LittleEndian, raw); err != nil {
					return nil, 0, fmt.Errorf("read pcm16: %w", err)
				}
				samples = make([]float32, nSamples)
				for i, v := range raw {
					samples[i] = float32(v) / 32768.0
				}
			case 24:
				nSamples := subSize / 3
				raw := make([]byte, subSize)
				if _, err := io.ReadFull(f, raw); err != nil {
					return nil, 0, fmt.Errorf("read pcm24: %w", err)
				}
				samples = make([]float32, nSamples)
				for i := 0; i < int(nSamples); i++ {
					// 24-bit signed little-endian: sign-extend from byte[2]
					lo := uint32(raw[i*3])
					mid := uint32(raw[i*3+1])
					hi := uint32(raw[i*3+2])
					val := int32(lo | mid<<8 | hi<<16)
					if val&0x800000 != 0 {
						val |= ^0xFFFFFF // sign extend
					}
					samples[i] = float32(val) / 8388608.0
				}
			case 32:
				nSamples := subSize / 4
				if audioFormat == 3 {
					// IEEE float32
					samples = make([]float32, nSamples)
					binary.Read(f, binary.LittleEndian, samples)
				} else {
					raw := make([]int32, nSamples)
					binary.Read(f, binary.LittleEndian, raw)
					samples = make([]float32, nSamples)
					for i, v := range raw {
						samples[i] = float32(v) / 2147483648.0
					}
				}
			default:
				return nil, 0, fmt.Errorf("unsupported bits per sample: %d", bitsPerSample)
			}

			// Stereo → mono
			if numChannels == 2 {
				mono := make([]float32, len(samples)/2)
				for i := range mono {
					mono[i] = (samples[i*2] + samples[i*2+1]) / 2.0
				}
				samples = mono
			} else if numChannels > 2 {
				return nil, 0, fmt.Errorf("unsupported channels: %d", numChannels)
			}

			// Resample to 16kHz if needed
			sr := int(sampleRate)
			if sr != 16000 {
				samples = resample(samples, sr, 16000)
				sr = 16000
			}

			return samples, sr, nil

		default:
			// Skip unknown chunks
			f.Seek(int64(subSize), io.SeekCurrent)
		}
	}

	if !dataFound {
		return nil, 0, fmt.Errorf("no data chunk found")
	}
	return nil, 0, fmt.Errorf("unexpected end of file")
}

// resample performs linear interpolation resampling.
func resample(samples []float32, fromRate, toRate int) []float32 {
	ratio := float64(fromRate) / float64(toRate)
	outLen := int(float64(len(samples)) / ratio)
	out := make([]float32, outLen)

	for i := 0; i < outLen; i++ {
		srcIdx := float64(i) * ratio
		idx := int(srcIdx)
		frac := float32(srcIdx - float64(idx))

		if idx+1 < len(samples) {
			out[i] = samples[idx]*(1-frac) + samples[idx+1]*frac
		} else if idx < len(samples) {
			out[i] = samples[idx]
		}
	}
	return out
}
