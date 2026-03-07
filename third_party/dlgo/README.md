# dlgo

Pure Go deep learning inference. Load GGUF models and run them on CPU with zero dependencies beyond the Go standard library.

```go
model, _ := dlgo.LoadLLM("model.gguf")
response, _ := model.Chat("", "What is the capital of France?")
fmt.Println(response) // "The capital of France is Paris."
```

## Features

- **LLM inference** — text generation, multi-turn chat, streaming
- **Speech-to-text** — Whisper transcription from WAV files
- **Voice activity detection** — Silero VAD
- **GGUF format** — loads quantized models directly, no conversion needed
- **Vulkan GPU inference** — full Vulkan compute backend with quantized MatVec shaders (Q4_0, Q4_K, Q5_0, Q6_K, Q8_0, F32), fused attention, RoPE, SwiGLU/GeGLU, RMSNorm — within 16–25% of Ollama GPU on generation
- **Fast on CPU** — AVX2/FMA/VNNI SIMD via optional CGo, QxQ integer dot products, batch prefill GEMM, parallel worker pools (within 7–17% of Ollama on generation, same GGUF)
- **25+ quantization formats** — Q4_0 through Q8_0, K-quants (Q2_K–Q8_K), I-quants, F16, BF16, F32

## Supported Architectures

| Architecture | Models Tested | Throughput |
|---|---|---|
| LLaMA | Llama 3.2 1B, TinyLlama 1.1B | ~60 tok/s |
| Qwen2/3 | Qwen 2.5 0.5B, Qwen3 0.6B | ~89 tok/s |
| Qwen3.5 | Qwen3.5 2B (hybrid GDN+attention) | ~16 tok/s |
| Gemma 2/3 | Gemma 2 2B, Gemma 3 1B, Gemma 3 270M | ~44–155 tok/s |
| SmolLM2 | SmolLM2 360M, SmolLM2 1.7B | ~39–97 tok/s |
| Phi | Phi-2, Phi-4-mini | ~8–9 tok/s |
| Mistral | Mistral (llama-compatible) | — |
| Whisper | Tiny, Base, Small (speech-to-text) | ~1x realtime |

Throughput measured with AVX2+FMA SIMD, parallel worker pool, batch prefill. CPU: AMD Ryzen 9 / Intel i9 class.

## Benchmarks vs Ollama (CPU-only)

Benchmarks use the **exact same GGUF file** loaded into both engines via `ollama create`
with a Modelfile. `temperature=0`, `seed=42`, `max_tokens=128`, Ollama forced CPU-only
(`num_gpu=0`). Three prompts averaged per model.

| Model | Quant | dlgo gen | Ollama gen | Delta | dlgo prefill | Ollama prefill |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | Q4_0 | 62.0 tok/s | 70.9 tok/s | −12% | 177 ms | 122 ms |
| Qwen 2.5 0.5B | Q4_K_M | 90.4 tok/s | 109.6 tok/s | −17% | 64 ms | 23 ms |
| Gemma 3 1B | Q4_K_M | 43.6 tok/s | 50.4 tok/s | −14% | 194 ms | 85 ms |
| SmolLM2 360M | Q8_0 | 97.4 tok/s | 111.1 tok/s | −12% | 52 ms | 37 ms |
| SmolLM2 1.7B | Q4_K_M | 38.6 tok/s | 42.7 tok/s | −10% | 188 ms | 130 ms |
| Gemma 3 270M | Q8_0 | 140.0 tok/s | 149.9 tok/s | −7% | 38 ms | 11 ms |

**Notes:**
- Generation gap (7–17%) is consistent across all models and quantizations, indicating
  the SIMD compute kernels are at parity with llama.cpp — the remaining cost is Go+CGo
  dispatch overhead (channel-based worker pool, goroutine scheduling, CGo call bridge).
- The gap shrinks on smaller models (270M: −7%) where per-token overhead is a larger
  fraction of total work.
- Ollama prefill times for prompts 2–3 benefit from KV cache reuse of the system prompt.
  Cold-start prefill (prompt 1) is closer: TinyLlama dlgo=172ms vs Ollama=184ms.

## Install

```bash
go get github.com/computerex/dlgo
```

## Usage

### Chat

```go
model, err := dlgo.LoadLLM("llama-3.2-1b-instruct-q4_k_m.gguf")
if err != nil {
    log.Fatal(err)
}

response, err := model.Chat(
    "You are a helpful assistant.",
    "Explain quantum computing in one sentence.",
    dlgo.WithMaxTokens(128),
    dlgo.WithTemperature(0.7),
)
fmt.Println(response)
```

### Streaming

```go
model, _ := dlgo.LoadLLM("model.gguf")

model.ChatStream("", "Write a poem about Go.", func(token string) {
    fmt.Print(token)
}, dlgo.WithMaxTokens(256))
```

### Multi-turn conversation

```go
response, _ := model.ChatMessages([]dlgo.Message{
    {Role: "system", Content: "You are a pirate."},
    {Role: "user", Content: "Tell me about the sea."},
    {Role: "assistant", Content: "Arrr, the sea be vast!"},
    {Role: "user", Content: "What about treasure?"},
}, dlgo.WithMaxTokens(128))
```

### Speech-to-text

```go
whisper, _ := dlgo.LoadWhisper("whisper-base.gguf", "tokenizer.json")
text, _ := whisper.TranscribeFile("audio.wav")
fmt.Println(text)
```

### Sampling options

```go
dlgo.WithMaxTokens(256)     // max tokens to generate
dlgo.WithTemperature(0.8)   // 0 = greedy, higher = more creative
dlgo.WithTopK(40)           // top-K sampling
dlgo.WithTopP(0.9)          // nucleus sampling
dlgo.WithGreedy()           // deterministic output
```

## Project Structure

```
dlgo.go          High-level API (LoadLLM, Chat, Generate, Stream)
core/            QuantizedTensor with row-level dequantization
quant/           25+ GGML quantization formats, fused SIMD dot products
format/gguf/     GGUF v2/v3 parser
format/ggml/     Legacy GGML parser
gpu/             Vulkan GPU compute backend (MatVec, attention, RoPE, etc.)
ops/             RMSNorm, RoPE, Softmax, SwiGLU, GeGLU, sampling
blas/            Quantized matrix-vector multiply, parallel worker pool
layers/          Conv1D, LSTM, GRU, MHA, GQA, cross-attention
audio/           WAV loading, STFT, mel spectrogram
memory/          KV cache, buffer pool
decode/          Greedy decode, beam search
models/llm/      LLM pipeline (tokenizer, forward, generation, chat templates)
models/whisper/  Whisper speech-to-text
models/silero/   Silero voice activity detection
examples/        Ready-to-run examples
```

## Quantization Guide

See **[docs/quantization-guide.md](docs/quantization-guide.md)** for detailed guidance on
choosing quantization formats. Summary:

| Tier | Types | Speed | Use Case |
|---|---|---|---|
| **Tier 1** (QxQ integer SIMD) | Q4_0, Q8_0, Q2_K–Q6_K, Q5_0 | Fastest | Recommended for all use |
| **Tier 2** (float SIMD) | F16, Q4_1, Q5_1 | 2–4x slower | Functional, avoid if possible |
| **Tier 3** (dequant+dot) | IQ*, TQ*, BF16 | Slowest | Avoid for large models |

All common GGUF downloads (Q4_K_M, Q5_K_M, Q3_K_L, Q8_0, etc.) use only Tier 1 types.

## GPU Benchmarks vs Ollama

GPU benchmarks on NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM). Same GGUF files loaded
into both engines. Greedy decoding, 8-token generation.

| Model | Quant | dlgo GPU gen | Ollama GPU gen | Delta |
|---|---|---|---|---|
| SmolLM2 360M | Q8_0 | ~420 tok/s | ~500 tok/s | −16% |
| TinyLlama 1.1B | Q4_0 | ~390 tok/s | ~490 tok/s | −20% |
| Qwen 2.5 0.5B | Q4_K_M | ~440 tok/s | ~575 tok/s | −24% |
| Gemma 3 1B | Q4_K_M | ~250 tok/s | ~325 tok/s | −23% |

**GPU implementation highlights:**
- Pure Vulkan compute (cross-platform, no CUDA dependency)
- Quantized MatVec shaders with typed struct access (`float16_t`, `uint8_t`, `int8_t`)
- Multi-warp per row for better SM utilization on small matrices
- Fused multi-head attention kernel (Q·K softmax, V accumulation)
- HOST_CACHED staging buffer for 14x faster CPU←GPU data transfer
- Single command buffer submission per token with batched dispatches
- Push descriptors (`VK_KHR_push_descriptor`) for minimal dispatch overhead

The 16–24% gap vs Ollama's CUDA backend is attributable to Vulkan vs CUDA dispatch overhead
and SPIR-V → PTX compilation quality vs native NVCC.

## How It Works

1. **GGUF parser** reads model metadata and tensor locations from the file
2. **Quantized tensors** stay in their compressed format in memory — only dequantized on the fly during matrix multiplication
3. **Forward pass** runs the model: embedding, RoPE, GQA attention, SwiGLU/GeGLU FFN, RMSNorm, and hybrid SSM/attention (Gated Delta Net) — architecture variations are expressed as a per-layer `LayerSpec` resolved at load time
4. **SIMD acceleration** (optional, via CGo) uses AVX2+FMA+VNNI for QxQ integer dot products and batch prefill GEMM kernels
5. **Parallel matmul** distributes rows across a persistent worker pool with fused multi-matrix dispatch
6. **GPU acceleration** (optional, via Vulkan) offloads the entire forward pass to GPU — all weights, KV cache, and intermediate buffers reside in VRAM
7. **Token sampling** supports temperature, top-K, top-P, min-P, and repetition penalty
