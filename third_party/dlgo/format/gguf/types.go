// Package gguf provides a parser for the GGUF (GPT-Generated Unified Format) model format.
//
// GGUF is a binary file format designed for storing large language models and other
// neural network weights in a compact, efficiently-loadable format. It supports:
//
// - Multiple quantization formats (F32, F16, BF16, Q4_0, Q8_0, Q2_K through Q8_K, IQ variants, etc.)
// - Extensible metadata via key-value pairs
// - Memory-mapped loading for zero-copy access
// - Multiple architectures (x86, ARM, etc.)
//
// The format consists of three main sections:
// 1. Header: Magic bytes, version, tensor count, metadata count
// 2. Key-Value Metadata: Model metadata like architecture, parameters, vocab size
// 3. Tensor Data: Binary tensor data with metadata (name, dimensions, type, offset)
//
// This package provides utilities for parsing GGUF files and extracting tensor
// information for use with quantization operations in the quant package.
package gguf

// GGUFType represents the type of a GGUF metadata value.
// These MUST match the GGUF specification exactly.
type GGUFType uint32

const (
	GGUFTypeUint8   GGUFType = 0
	GGUFTypeInt8    GGUFType = 1
	GGUFTypeUint16  GGUFType = 2
	GGUFTypeInt16   GGUFType = 3
	GGUFTypeUint32  GGUFType = 4
	GGUFTypeInt32   GGUFType = 5
	GGUFTypeFloat32 GGUFType = 6
	GGUFTypeBool    GGUFType = 7
	GGUFTypeString  GGUFType = 8
	GGUFTypeArray   GGUFType = 9
	GGUFTypeUint64  GGUFType = 10
	GGUFTypeInt64   GGUFType = 11
	GGUFTypeFloat64 GGUFType = 12
)

// GGMLType represents the quantization type of a tensor's data.
// These types correspond to the quantization formats supported by the quant package.
type GGMLType uint32

const (
	GGMLTypeF32     GGMLType = 0
	GGMLTypeF16     GGMLType = 1
	GGMLTypeQ4_0    GGMLType = 2
	GGMLTypeQ4_1    GGMLType = 3
	GGMLTypeQ5_0    GGMLType = 6
	GGMLTypeQ5_1    GGMLType = 7
	GGMLTypeQ8_0    GGMLType = 8
	GGMLTypeQ8_1    GGMLType = 9
	GGMLTypeQ2_K    GGMLType = 10
	GGMLTypeQ3_K    GGMLType = 11
	GGMLTypeQ4_K    GGMLType = 12
	GGMLTypeQ5_K    GGMLType = 13
	GGMLTypeQ6_K    GGMLType = 14
	GGMLTypeQ8_K    GGMLType = 15
	GGMLTypeIQ2_XXS GGMLType = 16
	GGMLTypeIQ2_XS  GGMLType = 17
	GGMLTypeIQ3_XXS GGMLType = 18
	GGMLTypeIQ1_S   GGMLType = 19
	GGMLTypeIQ4_NL  GGMLType = 20
	GGMLTypeIQ3_S   GGMLType = 21
	GGMLTypeIQ2_S   GGMLType = 22
	GGMLTypeIQ4_XS  GGMLType = 23
	GGMLTypeI8      GGMLType = 24
	GGMLTypeI16     GGMLType = 25
	GGMLTypeI32     GGMLType = 26
	GGMLTypeI64     GGMLType = 27
	GGMLTypeF64     GGMLType = 28
	GGMLTypeIQ1_M   GGMLType = 29
	GGMLTypeBF16    GGMLType = 30
	GGMLTypeTQ1_0   GGMLType = 34
	GGMLTypeTQ2_0   GGMLType = 35
)

// KV holds one metadata key-value pair from a GGUF file.
type KV struct {
	Key   string
	Type  GGUFType
	Value interface{} // actual Go type depends on Type
}

// TensorInfo holds metadata about one tensor in the GGUF file.
type TensorInfo struct {
	Name       string   // Tensor name (e.g., "blk.0.attn_q.weight")
	NDims      uint32   // Number of dimensions
	Dimensions []int64  // Shape (e.g., [2048, 32000] for embedding)
	Type       GGMLType // Quantization type (must match quant package types)
	Offset     uint64   // Offset relative to start of tensor data blob
}

// GGUFFile holds the parsed contents of a GGUF file.
//
// This struct provides access to all metadata and tensor information from a parsed
// GGUF model file. The tensor data itself is accessed via the file offset.
type GGUFFile struct {
	Version       uint32                 // GGUF format version
	TensorCount   int64                  // Number of tensors in the file
	MetadataCount int64                  // Number of key-value metadata pairs
	Metadata      map[string]interface{}  // Key-value metadata (model parameters, architecture info, etc.)
	Tensors       []TensorInfo           // Tensor metadata (name, shape, type, offset)
	DataOffset    int64                  // Absolute byte offset where tensor data begins
	FilePath      string                 // Path to the GGUF file (for reference)
}