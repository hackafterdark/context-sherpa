//go:build !amd64 || !cgo

package quant

// Q8BufferSize returns the number of bytes needed for Q8 quantization.
func Q8BufferSize(wType uint32, n int) int {
	switch wType {
	case 10, 11, 12, 13, 14:
		return (n / 256) * 292
	default:
		return (n / 32) * 34
	}
}

// QuantizeForType is a stub that does nothing on non-AVX2 platforms.
func QuantizeForType(x []float32, out []byte, wType uint32) {}

// QQDotBatch falls back to the float-based path on non-AVX2 platforms.
func QQDotBatch(wData []byte, wType uint32, qData []byte, cols int,
	out []float32, nrows int, bytesPerRow int) {
	for r := 0; r < nrows; r++ {
		rowData := wData[r*bytesPerRow : (r+1)*bytesPerRow]
		out[r] = FusedDotProduct(rowData, wType, nil, cols)
	}
}

// QQBatchGEMM falls back to per-position QQDotBatch on non-AVX2 platforms.
func QQBatchGEMM(wData []byte, wType uint32, q8Flat []byte, q8Stride int, nInputs int, cols int,
	outFlat []float32, nRows int, outStride int, bytesPerRow int) {
	for p := 0; p < nInputs; p++ {
		q8 := q8Flat[p*q8Stride : (p+1)*q8Stride]
		QQDotBatch(wData, wType, q8, cols, outFlat[p*outStride:p*outStride+nRows], nRows, bytesPerRow)
	}
}

// HasQQDot returns false on non-AVX2 platforms.
func HasQQDot(ggmlType uint32) bool {
	return false
}
