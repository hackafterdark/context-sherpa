//go:build !amd64 || !cgo

package quant

import "math"

func SIMDDotProduct(data []byte, ggmlType uint32, x []float32, n int) float32 {
	return FusedDotProduct(data, ggmlType, x, n)
}

func SIMDDotBatch(data []byte, ggmlType uint32, x []float32, cols int, out []float32, nrows int, bytesPerRow int) {
	if nrows <= 0 {
		return
	}
	for r := 0; r < nrows; r++ {
		rowData := data[r*bytesPerRow : (r+1)*bytesPerRow]
		out[r] = FusedDotProduct(rowData, ggmlType, x, cols)
	}
}

func SIMDDotF32(a, b []float32, n int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	limit := n - 7
	for ; i < limit; i += 8 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
		s4 += a[i+4] * b[i+4]
		s5 += a[i+5] * b[i+5]
		s6 += a[i+6] * b[i+6]
		s7 += a[i+7] * b[i+7]
	}
	sum := s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func SIMDDotF32Batch(aFlat []float32, x []float32, cols int, out []float32, nrows int) {
	for r := 0; r < nrows; r++ {
		out[r] = SIMDDotF32(aFlat[r*cols:(r+1)*cols], x, cols)
	}
}

func SIMDScaleAdd(out []float32, scale float32, src []float32, n int) {
	for i := 0; i < n; i++ {
		out[i] += scale * src[i]
	}
}

func SIMDSoftmax(x []float32) {
	// Fallback: use standard softmax
	if len(x) == 0 {
		return
	}
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	for i, v := range x {
		e := float32(math.Exp(float64(v - maxVal)))
		x[i] = e
		sum += e
	}
	for i := range x {
		x[i] /= sum
	}
}

func SIMDSwiGLU(out, gate, up []float32, n int) {
	for i := 0; i < n; i++ {
		g := gate[i]
		out[i] = (g / (1.0 + float32(math.Exp(float64(-g))))) * up[i]
	}
}

func HasSIMDDot(ggmlType uint32) bool {
	switch ggmlType {
	case 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14:
		return true
	default:
		return false
	}
}
