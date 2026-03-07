package ggml

import "fmt"

// WhisperTensorOrder defines the expected order of tensors in a Whisper GGML model.
//
// This is based on the whisper.cpp implementation and serves as a reference
// for validating GGML model files. Modern GGUF files use named tensors
// directly, making explicit ordering unnecessary.
var WhisperTensorOrder = []string{
	// Encoder globals
	"encoder.conv1.weight",
	"encoder.conv1.bias",
	"encoder.conv2.weight",
	"encoder.conv2.bias",
	"encoder.position_embedding.weight",
	"encoder.ln.weight",
	"encoder.ln.bias",

	// Encoder layers (repeated for each layer)
	// Layer 0
	"encoder.layers.0.attn.q_proj.weight",
	"encoder.layers.0.attn.k_proj.weight",
	"encoder.layers.0.attn.v_proj.weight",
	"encoder.layers.0.attn.out_proj.weight",
	"encoder.layers.0.attn.out_proj.bias",
	"encoder.layers.0.attn.ln.weight",
	"encoder.layers.0.attn.ln.bias",
	"encoder.layers.0.ffn_gate.weight",
	"encoder.layers.0.ffn_up.weight",
	"encoder.layers.0.ffn_down.weight",
	"encoder.layers.0.ffn.ln.weight",
	"encoder.layers.0.ffn.ln.bias",
}

// GetTensorName returns the expected tensor name for a given position index.
//
// For Whisper models, this uses a fixed ordering based on the model architecture.
// This function is primarily useful for legacy GGML models where tensor names
// are not explicitly stored in the file.
//
// Parameters:
//   - index: Tensor position index in the model
//   - config: Model configuration (e.g., number of encoder/decoder layers)
//
// Returns the expected tensor name, or empty string if index is out of range.
//
// Note: Modern GGUF files store tensor names directly, making this function
// unnecessary for new models.
func GetTensorName(index int, config map[string]int32) string {
	// This is a simplified version - in a full implementation,
	// we would generate the complete tensor order and look up by index.
	// For now, return a placeholder as the original code does.
	return ""
}

// GenerateTensorOrder generates the complete list of expected tensor names
// based on the model configuration.
//
// This function is used to validate that all required tensors are present
// in a GGML model file. It generates tensor names for both encoder
// and decoder layers based on the specified layer counts.
//
// Parameters:
//   - nEncLayers: Number of encoder layers
//   - nDecLayers: Number of decoder layers
//
// Returns a slice of expected tensor names in the order they should appear.
//
// Example usage:
//
//	// Generate tensor order for Whisper tiny model (4 encoder, 4 decoder layers)
//	tensorOrder := GenerateTensorOrder(4, 4)
//
//	// Validate that all tensors are present
//	for _, expectedName := range tensorOrder {
//	    if _, exists := model.Tensors[expectedName]; !exists {
//	        return fmt.Errorf("missing tensor: %s", expectedName)
//	    }
//	}
func GenerateTensorOrder(nEncLayers, nDecLayers int) []string {
	var order []string

	// Encoder globals
	order = append(order, "encoder.conv1.weight")
	order = append(order, "encoder.conv1.bias")
	order = append(order, "encoder.conv2.weight")
	order = append(order, "encoder.conv2.bias")
	order = append(order, "encoder.position_embedding.weight")
	order = append(order, "encoder.ln.weight")
	order = append(order, "encoder.ln.bias")

	// Encoder layers
	for i := 0; i < nEncLayers; i++ {
		order = append(order, fmt.Sprintf("encoder.layers.%d.attn.q_proj.weight", i))
		order = append(order, fmt.Sprintf("encoder.layers.%d.attn.k_proj.weight", i))
		order = append(order, fmt.Sprintf("encoder.layers.%d.attn.v_proj.weight", i))
		order = append(order, fmt.Sprintf("encoder.layers.%d.attn.out_proj.weight", i))
		order = append(order, fmt.Sprintf("encoder.layers.%d.attn.out_proj.bias", i))
		order = append(order, fmt.Sprintf("encoder.layers.%d.attn.ln.weight", i))
		order = append(order, fmt.Sprintf("encoder.layers.%d.attn.ln.bias", i))
		order = append(order, fmt.Sprintf("encoder.layers.%d.ffn_gate.weight", i))
		order = append(order, fmt.Sprintf("encoder.layers.%d.ffn_up.weight", i))
		order = append(order, fmt.Sprintf("encoder.layers.%d.ffn_down.weight", i))
		order = append(order, fmt.Sprintf("encoder.layers.%d.ffn.ln.weight", i))
		order = append(order, fmt.Sprintf("encoder.layers.%d.ffn.ln.bias", i))
	}

	// Decoder globals
	order = append(order, "decoder.token_embedding.weight")
	order = append(order, "decoder.position_embedding.weight")
	order = append(order, "decoder.ln.weight")
	order = append(order, "decoder.ln.bias")
	order = append(order, "decoder.proj.weight")

	// Decoder layers
	for i := 0; i < nDecLayers; i++ {
		order = append(order, fmt.Sprintf("decoder.layers.%d.attn.q_proj.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.attn.k_proj.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.attn.v_proj.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.attn.out_proj.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.attn.out_proj.bias", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.attn.ln.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.attn.ln.bias", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.cross_attn.q_proj.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.cross_attn.k_proj.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.cross_attn.v_proj.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.cross_attn.out_proj.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.cross_attn.out_proj.bias", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.cross_attn.ln.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.cross_attn.ln.bias", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.ffn_gate.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.ffn_up.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.ffn_down.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.ffn.ln.weight", i))
		order = append(order, fmt.Sprintf("decoder.layers.%d.ffn.ln.bias", i))
	}

	return order
}