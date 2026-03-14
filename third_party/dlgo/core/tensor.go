package core

import (
	"fmt"

	"github.com/computerex/dlgo/quant"
)

// QuantizedTensor stores weights in their original quantized format with
// optional pre-dequantized FP32 cache.
type QuantizedTensor struct {
	Data     []byte    // Raw quantized bytes
	Type     uint32    // GGML quantization type
	Rows     int       // Outer dimension
	Cols     int       // Inner dimension
	FP32Data []float32 // Pre-dequantized FP32 cache
}

// NewQuantizedTensor creates a new QuantizedTensor with validation.
func NewQuantizedTensor(data []byte, ggmlType uint32, rows, cols int) (*QuantizedTensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("invalid dimensions: rows=%d cols=%d", rows, cols)
	}

	totalElements := rows * cols
	expectedBytes := quant.BytesForN(ggmlType, totalElements)
	if len(data) != expectedBytes {
		return nil, fmt.Errorf("data length mismatch: expected %d, got %d", expectedBytes, len(data))
	}

	return &QuantizedTensor{
		Data: data,
		Type: ggmlType,
		Rows: rows,
		Cols: cols,
	}, nil
}

// DequantizeRow dequantizes a single row into the provided buffer.
func (qt *QuantizedTensor) DequantizeRow(rowIdx int, out []float32) error {
	if rowIdx < 0 || rowIdx >= qt.Rows {
		return fmt.Errorf("row index out of bounds: %d", rowIdx)
	}
	if len(out) < qt.Cols {
		return fmt.Errorf("output buffer too small: need %d, got %d", qt.Cols, len(out))
	}

	if qt.FP32Data != nil {
		copy(out, qt.FP32Data[rowIdx*qt.Cols:(rowIdx+1)*qt.Cols])
		return nil
	}

	bytesPerRow := quant.BytesForN(qt.Type, qt.Cols)
	rowData := qt.Data[rowIdx*bytesPerRow : (rowIdx+1)*bytesPerRow]
	quant.DequantizeInto(out, rowData, qt.Type, qt.Cols)
	return nil
}

// Close releases all resources associated with the tensor.
func (qt *QuantizedTensor) Close() error {
	qt.Data = nil
	qt.FP32Data = nil
	return nil
}