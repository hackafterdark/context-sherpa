package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// countingReader tracks the number of bytes read from an io.Reader.
// This is critical for computing tensor data offsets in GGUF files.
type countingReader struct {
	r     io.Reader
	count int64
}

func (cr *countingReader) Read(p []byte) (int, error) {
	n, err := cr.r.Read(p)
	cr.count += int64(n)
	return n, err
}

// readString reads a GGUF string format (uint64 length prefix + UTF-8 bytes).
func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readValue reads a GGUF value based on its type.
// Returns the value as a Go interface{} where the actual type depends on the GGUFType.
func readValue(r io.Reader, valueType GGUFType) (interface{}, error) {
	switch valueType {
	case GGUFTypeUint8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUFTypeInt8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUFTypeUint16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUFTypeInt16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUFTypeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUFTypeInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUFTypeFloat32:
		var bits uint32
		if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
			return nil, err
		}
		return math.Float32frombits(bits), nil
	case GGUFTypeBool:
		var v int8
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		return v != 0, nil
	case GGUFTypeString:
		return readString(r)
	case GGUFTypeUint64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUFTypeInt64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUFTypeFloat64:
		var bits uint64
		if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
			return nil, err
		}
		return math.Float64frombits(bits), nil
	case GGUFTypeArray:
		// Read element type
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, err
		}
		// Read array length
		var length uint64
		if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
			return nil, err
		}
		// Read elements
		arr := make([]interface{}, length)
		for i := uint64(0); i < length; i++ {
			val, err := readValue(r, GGUFType(elemType))
			if err != nil {
				return nil, err
			}
			arr[i] = val
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unsupported GGUF type: %d", valueType)
	}
}

// Open opens and parses a GGUF file from disk.
//
// This function reads the GGUF file and extracts all metadata and tensor information.
// The tensor data itself is not loaded - it can be accessed via the DataOffset field
// for memory-mapped access.
//
// Supported GGUF versions: 2 and 3
//
// Parameters:
//   - filePath: Path to the GGUF model file
//
// Returns a GGUFFile struct containing all parsed metadata and tensor information.
func Open(filePath string) (*GGUFFile, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// Create counting reader to track byte offsets
	counter := &countingReader{r: bufio.NewReader(file)}

	// Read and verify magic bytes
	magic := make([]byte, 4)
	if _, err := io.ReadFull(counter, magic); err != nil {
		return nil, fmt.Errorf("failed to read magic: %w", err)
	}
	if string(magic) != "GGUF" {
		return nil, fmt.Errorf("invalid magic: expected GGUF, got %s", string(magic))
	}

	// Read GGUF version
	var version uint32
	if err := binary.Read(counter, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("failed to read version: %w", err)
	}
	if version != 2 && version != 3 {
		return nil, fmt.Errorf("unsupported version: %d (expected 2 or 3)", version)
	}

	// Read tensor count
	var tensorCount int64
	if err := binary.Read(counter, binary.LittleEndian, &tensorCount); err != nil {
		return nil, fmt.Errorf("failed to read tensor count: %w", err)
	}

	// Read key-value metadata count
	var kvCount int64
	if err := binary.Read(counter, binary.LittleEndian, &kvCount); err != nil {
		return nil, fmt.Errorf("failed to read KV count: %w", err)
	}

	// Read all key-value metadata pairs
	metadata := make(map[string]interface{})
	for i := int64(0); i < kvCount; i++ {
		// Read key name
		key, err := readString(counter)
		if err != nil {
			return nil, fmt.Errorf("failed to read KV key %d: %w", i, err)
		}

		// Read value type
		var valueType uint32
		if err := binary.Read(counter, binary.LittleEndian, &valueType); err != nil {
			return nil, fmt.Errorf("failed to read KV value type for key %s: %w", key, err)
		}

		// Read value
		value, err := readValue(counter, GGUFType(valueType))
		if err != nil {
			return nil, fmt.Errorf("failed to read KV value for key %s: %w", key, err)
		}

		metadata[key] = value
	}

	// Read tensor information for all tensors
	tensors := make([]TensorInfo, tensorCount)
	for i := int64(0); i < tensorCount; i++ {
		// Read tensor name
		name, err := readString(counter)
		if err != nil {
			return nil, fmt.Errorf("failed to read tensor name %d: %w", i, err)
		}

		// Read number of dimensions
		var nDims uint32
		if err := binary.Read(counter, binary.LittleEndian, &nDims); err != nil {
			return nil, fmt.Errorf("failed to read nDims for tensor %s: %w", name, err)
		}

		// Read dimensions
		dims := make([]int64, nDims)
		for d := uint32(0); d < nDims; d++ {
			if err := binary.Read(counter, binary.LittleEndian, &dims[d]); err != nil {
				return nil, fmt.Errorf("failed to read dimension %d for tensor %s: %w", d, name, err)
			}
		}

		// Read tensor type (quantization format)
		var tensorType uint32
		if err := binary.Read(counter, binary.LittleEndian, &tensorType); err != nil {
			return nil, fmt.Errorf("failed to read type for tensor %s: %w", name, err)
		}

		// Read tensor data offset
		var offset uint64
		if err := binary.Read(counter, binary.LittleEndian, &offset); err != nil {
			return nil, fmt.Errorf("failed to read offset for tensor %s: %w", name, err)
		}

		tensors[i] = TensorInfo{
			Name:       name,
			NDims:      nDims,
			Dimensions: dims,
			Type:       GGMLType(tensorType),
			Offset:     offset,
		}
	}

	// Compute alignment padding
	// GGUF pads the end of the header to align tensor data
	alignment := uint32(32) // default alignment
	if alignVal, ok := metadata["general.alignment"]; ok {
		if alignUint32, ok := alignVal.(uint32); ok {
			alignment = alignUint32
		}
	}

	// Skip alignment padding
	currentPos := counter.count
	padding := currentPos % int64(alignment)
	if padding != 0 {
		skipBytes := int64(alignment) - padding
		buf := make([]byte, skipBytes)
		if _, err := io.ReadFull(counter, buf); err != nil {
			return nil, fmt.Errorf("failed to skip padding: %w", err)
		}
	}

	// The current position is where tensor data begins
	dataOffset := counter.count

	return &GGUFFile{
		Version:       version,
		TensorCount:   tensorCount,
		MetadataCount: kvCount,
		Metadata:      metadata,
		Tensors:       tensors,
		DataOffset:    dataOffset,
		FilePath:      filePath,
	}, nil
}