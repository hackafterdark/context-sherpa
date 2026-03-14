// Command generate creates small test model files for the dlgo test suite.
//
// Usage: go run ./testdata/generate.go
//
// This generates:
//   - testdata/silero_tiny.ggml    — a minimal Silero VAD model (random weights)
//   - testdata/test.gguf           — a minimal GGUF file with F32 tensors
//   - testdata/test_quant.gguf     — a GGUF file with Q8_0 quantized tensors
//   - testdata/test_multi.gguf     — a GGUF file with mixed quantization types
//   - testdata/test_audio.wav      — a synthetic 16kHz mono WAV test file
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
)

func main() {
	models := []struct {
		name string
		gen  func(string) error
	}{
		{"testdata/silero_tiny.ggml", generateSileroGGML},
		{"testdata/test.gguf", generateTestGGUF},
		{"testdata/test_quant.gguf", generateQuantGGUF},
		{"testdata/test_multi.gguf", generateMultiQuantGGUF},
		{"testdata/test_audio.wav", generateTestWAV},
	}

	for _, m := range models {
		if err := m.gen(m.name); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", m.name, err)
			os.Exit(1)
		}
		fi, _ := os.Stat(m.name)
		fmt.Printf("wrote %s (%d bytes)\n", m.name, fi.Size())
	}
}

// generateSileroGGML writes a minimal Silero VAD model in the custom GGML format
// that models/silero/model.go LoadModel() expects.
func generateSileroGGML(path string) error {
	rng := rand.New(rand.NewSource(42))
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Magic: "ggml" little-endian = 0x67676d6c
	binary.Write(f, binary.LittleEndian, uint32(0x67676d6c))

	// Model type string
	modelType := "silero_vad"
	binary.Write(f, binary.LittleEndian, int32(len(modelType)))
	f.Write([]byte(modelType))

	// Version 5.0.0
	binary.Write(f, binary.LittleEndian, int32(5))
	binary.Write(f, binary.LittleEndian, int32(0))
	binary.Write(f, binary.LittleEndian, int32(0))

	// Architecture params
	binary.Write(f, binary.LittleEndian, int32(512))  // WindowSize
	binary.Write(f, binary.LittleEndian, int32(64))   // ContextSize
	binary.Write(f, binary.LittleEndian, int32(4))    // NEncoderLayers

	// Encoder layer configs: [inCh, outCh, kernelSize] × 4
	layerConfigs := [][3]int32{
		{129, 128, 3},
		{128, 64, 3},
		{64, 64, 3},
		{64, 128, 3},
	}
	for _, lc := range layerConfigs {
		binary.Write(f, binary.LittleEndian, lc[0])
		binary.Write(f, binary.LittleEndian, lc[1])
		binary.Write(f, binary.LittleEndian, lc[2])
	}

	binary.Write(f, binary.LittleEndian, int32(128)) // LSTMInputSize
	binary.Write(f, binary.LittleEndian, int32(128)) // LSTMHiddenSize
	binary.Write(f, binary.LittleEndian, int32(128)) // FinalConvIn
	binary.Write(f, binary.LittleEndian, int32(1))   // FinalConvOut

	// Helper to write one tensor
	writeTensor := func(name string, dims []int32, data []float32) {
		binary.Write(f, binary.LittleEndian, int32(len(dims)))
		binary.Write(f, binary.LittleEndian, int32(len(name)))
		binary.Write(f, binary.LittleEndian, int32(0)) // ftype=0 (f32)
		for _, d := range dims {
			binary.Write(f, binary.LittleEndian, d)
		}
		f.Write([]byte(name))
		binary.Write(f, binary.LittleEndian, data)
	}

	randFloats := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = (rng.Float32() - 0.5) * 0.1
		}
		return out
	}

	// STFT basis: [258, 1, 256] → 66048 elements
	writeTensor("_model.stft.forward_basis_buffer",
		[]int32{258, 1, 256},
		randFloats(258*256))

	// Encoder layers
	for i, lc := range layerConfigs {
		inCh, outCh, k := int(lc[0]), int(lc[1]), int(lc[2])
		writeTensor(fmt.Sprintf("_model.encoder.%d.reparam_conv.weight", i),
			[]int32{int32(outCh), int32(inCh), int32(k)},
			randFloats(outCh*inCh*k))
		writeTensor(fmt.Sprintf("_model.encoder.%d.reparam_conv.bias", i),
			[]int32{int32(outCh)},
			randFloats(outCh))
	}

	// LSTM weights
	writeTensor("_model.decoder.rnn.weight_ih",
		[]int32{512, 128},
		randFloats(512*128))
	writeTensor("_model.decoder.rnn.weight_hh",
		[]int32{512, 128},
		randFloats(512*128))
	writeTensor("_model.decoder.rnn.bias_ih",
		[]int32{512},
		randFloats(512))
	writeTensor("_model.decoder.rnn.bias_hh",
		[]int32{512},
		randFloats(512))

	// Final conv
	writeTensor("_model.decoder.decoder.2.weight",
		[]int32{1, 128, 1},
		randFloats(128))
	writeTensor("_model.decoder.decoder.2.bias",
		[]int32{1},
		randFloats(1))

	return nil
}

// generateTestGGUF writes a minimal GGUF v3 file with two small F32 tensors.
func generateTestGGUF(path string) error {
	rng := rand.New(rand.NewSource(99))
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Magic: "GGUF"
	f.Write([]byte("GGUF"))

	// Version 3
	binary.Write(f, binary.LittleEndian, uint32(3))

	// Tensor count: 2
	binary.Write(f, binary.LittleEndian, int64(2))

	// KV count: 3
	binary.Write(f, binary.LittleEndian, int64(3))

	// --- KV pairs ---

	writeGGUFString := func(s string) {
		binary.Write(f, binary.LittleEndian, uint64(len(s)))
		f.Write([]byte(s))
	}

	// KV 1: general.architecture = "test"
	writeGGUFString("general.architecture")
	binary.Write(f, binary.LittleEndian, uint32(8)) // string type
	writeGGUFString("test")

	// KV 2: general.name = "dlgo-test-model"
	writeGGUFString("general.name")
	binary.Write(f, binary.LittleEndian, uint32(8))
	writeGGUFString("dlgo-test-model")

	// KV 3: test.hidden_size = uint32(64)
	writeGGUFString("test.hidden_size")
	binary.Write(f, binary.LittleEndian, uint32(4)) // uint32 type
	binary.Write(f, binary.LittleEndian, uint32(64))

	// --- Tensor infos ---
	// Tensor 1: "weight_a" - [64, 32] F32
	writeGGUFString("weight_a")
	binary.Write(f, binary.LittleEndian, uint32(2)) // 2 dims
	binary.Write(f, binary.LittleEndian, int64(64))
	binary.Write(f, binary.LittleEndian, int64(32))
	binary.Write(f, binary.LittleEndian, uint32(0)) // F32
	binary.Write(f, binary.LittleEndian, uint64(0)) // offset 0

	tensor1Size := 64 * 32 * 4 // 8192 bytes

	// Tensor 2: "bias_a" - [64] F32
	writeGGUFString("bias_a")
	binary.Write(f, binary.LittleEndian, uint32(1)) // 1 dim
	binary.Write(f, binary.LittleEndian, int64(64))
	binary.Write(f, binary.LittleEndian, uint32(0)) // F32
	binary.Write(f, binary.LittleEndian, uint64(tensor1Size))

	// --- Alignment padding ---
	// Default alignment is 32 bytes
	currentPos := getCurrentPos(f)
	alignment := int64(32)
	remainder := currentPos % alignment
	if remainder != 0 {
		padSize := alignment - remainder
		padding := make([]byte, padSize)
		f.Write(padding)
	}

	// --- Tensor data ---
	// weight_a: 64*32 = 2048 floats
	for i := 0; i < 64*32; i++ {
		binary.Write(f, binary.LittleEndian, (rng.Float32()-0.5)*0.1)
	}
	// bias_a: 64 floats
	for i := 0; i < 64; i++ {
		binary.Write(f, binary.LittleEndian, float32(0.0))
	}

	return nil
}

func getCurrentPos(f *os.File) int64 {
	pos, _ := f.Seek(0, 1) // SeekCurrent
	return pos
}

// float32ToF16Bits converts a float32 to IEEE 754 half-precision bits.
func float32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 1
	exp := int((bits>>23)&0xff) - 127
	mant := bits & 0x7fffff

	if exp < -24 {
		return uint16(sign << 15)
	}
	if exp < -14 {
		mant |= 0x800000
		shift := uint(-14 - exp)
		mant >>= shift
		return uint16(sign<<15) | uint16(mant>>13)
	}
	if exp > 15 {
		return uint16(sign<<15) | 0x7C00
	}
	return uint16(sign<<15) | uint16((exp+15)<<10) | uint16(mant>>13)
}

// quantizeQ8_0 quantizes n float32 values into Q8_0 format.
// Q8_0: 32 values per block, 34 bytes per block (2 byte f16 scale + 32 int8 quants).
func quantizeQ8_0(values []float32) []byte {
	n := len(values)
	numBlocks := n / 32
	out := make([]byte, numBlocks*34)

	for b := 0; b < numBlocks; b++ {
		block := values[b*32 : (b+1)*32]

		var amax float32
		for _, v := range block {
			av := v
			if av < 0 {
				av = -av
			}
			if av > amax {
				amax = av
			}
		}

		d := amax / 127.0
		if d == 0 {
			d = 1.0
		}
		id := 1.0 / d

		off := b * 34
		binary.LittleEndian.PutUint16(out[off:], float32ToF16Bits(d))

		for j := 0; j < 32; j++ {
			q := int(math.Round(float64(block[j]) * float64(id)))
			if q > 127 {
				q = 127
			} else if q < -128 {
				q = -128
			}
			out[off+2+j] = byte(int8(q))
		}
	}
	return out
}

// quantizeQ4_0 quantizes n float32 values into Q4_0 format.
// Q4_0: 32 values per block, 18 bytes per block (2 byte f16 scale + 16 nibble bytes).
func quantizeQ4_0(values []float32) []byte {
	n := len(values)
	numBlocks := n / 32
	out := make([]byte, numBlocks*18)

	for b := 0; b < numBlocks; b++ {
		block := values[b*32 : (b+1)*32]

		var amax float32
		for _, v := range block {
			av := v
			if av < 0 {
				av = -av
			}
			if av > amax {
				amax = av
			}
		}

		d := amax / 7.0
		if d == 0 {
			d = 1.0
		}
		id := 1.0 / d

		off := b * 18
		binary.LittleEndian.PutUint16(out[off:], float32ToF16Bits(d))

		for j := 0; j < 16; j++ {
			q0 := int(math.Round(float64(block[j])*float64(id))) + 8
			q1 := int(math.Round(float64(block[j+16])*float64(id))) + 8
			if q0 < 0 {
				q0 = 0
			} else if q0 > 15 {
				q0 = 15
			}
			if q1 < 0 {
				q1 = 0
			} else if q1 > 15 {
				q1 = 15
			}
			out[off+2+j] = byte(q0) | byte(q1<<4)
		}
	}
	return out
}

// generateQuantGGUF writes a GGUF v3 file with Q8_0 quantized tensors.
func generateQuantGGUF(path string) error {
	rng := rand.New(rand.NewSource(77))
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	f.Write([]byte("GGUF"))
	binary.Write(f, binary.LittleEndian, uint32(3))  // version
	binary.Write(f, binary.LittleEndian, int64(2))   // tensor count
	binary.Write(f, binary.LittleEndian, int64(2))   // kv count

	writeGGUFString := func(s string) {
		binary.Write(f, binary.LittleEndian, uint64(len(s)))
		f.Write([]byte(s))
	}

	// KV 1
	writeGGUFString("general.architecture")
	binary.Write(f, binary.LittleEndian, uint32(8))
	writeGGUFString("test-quant")

	// KV 2
	writeGGUFString("general.name")
	binary.Write(f, binary.LittleEndian, uint32(8))
	writeGGUFString("dlgo-q8-test")

	// Tensor 1: "layer.0.weight" - [128, 32] Q8_0
	writeGGUFString("layer.0.weight")
	binary.Write(f, binary.LittleEndian, uint32(2)) // 2 dims
	binary.Write(f, binary.LittleEndian, int64(128))
	binary.Write(f, binary.LittleEndian, int64(32))
	binary.Write(f, binary.LittleEndian, uint32(8)) // Q8_0
	binary.Write(f, binary.LittleEndian, uint64(0))

	// 128*32 = 4096 elements, Q8_0 = (4096/32)*34 = 4352 bytes
	t1Vals := make([]float32, 128*32)
	for i := range t1Vals {
		t1Vals[i] = (rng.Float32() - 0.5) * 2.0
	}
	t1Data := quantizeQ8_0(t1Vals)

	// Tensor 2: "layer.0.bias" - [128] F32
	writeGGUFString("layer.0.bias")
	binary.Write(f, binary.LittleEndian, uint32(1)) // 1 dim
	binary.Write(f, binary.LittleEndian, int64(128))
	binary.Write(f, binary.LittleEndian, uint32(0)) // F32
	binary.Write(f, binary.LittleEndian, uint64(len(t1Data)))

	// Alignment padding
	currentPos := getCurrentPos(f)
	alignment := int64(32)
	remainder := currentPos % alignment
	if remainder != 0 {
		f.Write(make([]byte, alignment-remainder))
	}

	// Tensor data
	f.Write(t1Data)
	for i := 0; i < 128; i++ {
		binary.Write(f, binary.LittleEndian, float32(0.0))
	}

	return nil
}

// generateMultiQuantGGUF writes a GGUF file with mixed F32, F16, Q4_0, and Q8_0 tensors.
func generateMultiQuantGGUF(path string) error {
	rng := rand.New(rand.NewSource(55))
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	f.Write([]byte("GGUF"))
	binary.Write(f, binary.LittleEndian, uint32(3))
	binary.Write(f, binary.LittleEndian, int64(4))  // 4 tensors
	binary.Write(f, binary.LittleEndian, int64(3))  // 3 KVs

	writeGGUFString := func(s string) {
		binary.Write(f, binary.LittleEndian, uint64(len(s)))
		f.Write([]byte(s))
	}

	writeGGUFString("general.architecture")
	binary.Write(f, binary.LittleEndian, uint32(8))
	writeGGUFString("test-multi")

	writeGGUFString("general.name")
	binary.Write(f, binary.LittleEndian, uint32(8))
	writeGGUFString("dlgo-multi-quant")

	writeGGUFString("test.layer_count")
	binary.Write(f, binary.LittleEndian, uint32(4)) // uint32
	binary.Write(f, binary.LittleEndian, uint32(4))

	// Generate tensor data first to know offsets
	n32 := 64
	f32Vals := make([]float32, n32)
	for i := range f32Vals {
		f32Vals[i] = (rng.Float32() - 0.5) * 0.5
	}
	f32Data := make([]byte, n32*4)
	for i, v := range f32Vals {
		binary.LittleEndian.PutUint32(f32Data[i*4:], math.Float32bits(v))
	}

	n16 := 64
	f16Vals := make([]float32, n16)
	for i := range f16Vals {
		f16Vals[i] = (rng.Float32() - 0.5) * 0.5
	}
	f16Data := make([]byte, n16*2)
	for i, v := range f16Vals {
		binary.LittleEndian.PutUint16(f16Data[i*2:], float32ToF16Bits(v))
	}

	nQ4 := 256
	q4Vals := make([]float32, nQ4)
	for i := range q4Vals {
		q4Vals[i] = (rng.Float32() - 0.5) * 2.0
	}
	q4Data := quantizeQ4_0(q4Vals)

	nQ8 := 256
	q8Vals := make([]float32, nQ8)
	for i := range q8Vals {
		q8Vals[i] = (rng.Float32() - 0.5) * 2.0
	}
	q8Data := quantizeQ8_0(q8Vals)

	offset := uint64(0)

	// Tensor info: embed.weight [64] F32
	writeGGUFString("embed.weight")
	binary.Write(f, binary.LittleEndian, uint32(1))
	binary.Write(f, binary.LittleEndian, int64(64))
	binary.Write(f, binary.LittleEndian, uint32(0)) // F32
	binary.Write(f, binary.LittleEndian, offset)
	offset += uint64(len(f32Data))

	// Tensor info: embed.bias [64] F16
	writeGGUFString("embed.bias")
	binary.Write(f, binary.LittleEndian, uint32(1))
	binary.Write(f, binary.LittleEndian, int64(64))
	binary.Write(f, binary.LittleEndian, uint32(1)) // F16
	binary.Write(f, binary.LittleEndian, offset)
	offset += uint64(len(f16Data))

	// Tensor info: attn.weight [256] Q4_0
	writeGGUFString("attn.weight")
	binary.Write(f, binary.LittleEndian, uint32(1))
	binary.Write(f, binary.LittleEndian, int64(256))
	binary.Write(f, binary.LittleEndian, uint32(2)) // Q4_0
	binary.Write(f, binary.LittleEndian, offset)
	offset += uint64(len(q4Data))

	// Tensor info: ffn.weight [256] Q8_0
	writeGGUFString("ffn.weight")
	binary.Write(f, binary.LittleEndian, uint32(1))
	binary.Write(f, binary.LittleEndian, int64(256))
	binary.Write(f, binary.LittleEndian, uint32(8)) // Q8_0
	binary.Write(f, binary.LittleEndian, offset)

	// Alignment padding
	currentPos := getCurrentPos(f)
	alignment := int64(32)
	remainder := currentPos % alignment
	if remainder != 0 {
		f.Write(make([]byte, alignment-remainder))
	}

	// Write all tensor data sequentially
	f.Write(f32Data)
	f.Write(f16Data)
	f.Write(q4Data)
	f.Write(q8Data)

	return nil
}

// generateTestWAV creates a synthetic 16kHz mono 16-bit WAV file with a sine tone.
func generateTestWAV(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	sampleRate := uint32(16000)
	numSamples := 16000 // 1 second
	bitsPerSample := uint16(16)
	numChannels := uint16(1)
	bytesPerSample := bitsPerSample / 8
	dataSize := uint32(numSamples) * uint32(numChannels) * uint32(bytesPerSample)

	// RIFF header
	f.Write([]byte("RIFF"))
	binary.Write(f, binary.LittleEndian, uint32(36+dataSize))
	f.Write([]byte("WAVE"))

	// fmt chunk
	f.Write([]byte("fmt "))
	binary.Write(f, binary.LittleEndian, uint32(16))
	binary.Write(f, binary.LittleEndian, uint16(1)) // PCM
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

	// 440Hz sine tone at 0.5 amplitude
	for i := 0; i < numSamples; i++ {
		val := math.Sin(2.0 * math.Pi * 440.0 * float64(i) / float64(sampleRate))
		sample := int16(val * 16384) // 50% amplitude
		binary.Write(f, binary.LittleEndian, sample)
	}

	return nil
}
