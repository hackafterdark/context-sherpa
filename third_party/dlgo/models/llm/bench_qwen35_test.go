package llm

import (
	"os"
	"testing"
)

func TestBenchQwen35(t *testing.T) {
	path := `C:\projects\gollm\Qwen3.5-2B.Q4_K_M.gguf`
	if _, err := os.Stat(path); err != nil {
		t.Skip("model not found")
	}

	pipe, err := NewPipeline(path, 512)
	if err != nil {
		t.Fatalf("NewPipeline: %v", err)
	}

	mc := pipe.Model.Config
	t.Logf("Config: arch=%s dim=%d heads=%d/%d headDim=%d ropeDim=%d ffn=%d layers=%d vocab=%d",
		mc.Architecture, mc.EmbeddingDim, mc.NumHeads, mc.NumKVHeads,
		mc.HeadDim, mc.RopeDim, mc.FFNDim, mc.NumLayers, mc.VocabSize)
	t.Logf("SSM: interval=%d convK=%d inner=%d state=%d dt_rank=%d groups=%d",
		mc.FullAttentionInterval, mc.SSMConvKernel, mc.SSMInnerSize,
		mc.SSMStateSize, mc.SSMTimeStepRank, mc.SSMGroupCount)

	l0 := pipe.Model.Layers[0]
	t.Logf("Layer 0 (SSM): SSMInProj=%v AttnGate=%v Conv1dW=%d SSMA=%d SSMAlpha=%v SSMBeta=%v SSMOut=%v SSMNorm=%d",
		l0.SSMInProj != nil, l0.AttnGate != nil, len(l0.SSMConv1dW), len(l0.SSMA),
		l0.SSMAlpha != nil, l0.SSMBeta != nil, l0.SSMOut != nil, len(l0.SSMNorm))
	t.Logf("Layer 0 (FFN): FFNGate=%v FFNUp=%v FFNDown=%v", l0.FFNGate != nil, l0.FFNUp != nil, l0.FFNDown != nil)

	l3 := pipe.Model.Layers[3]
	t.Logf("Layer 3 (Attn): Wq=%v Wk=%v Wv=%v Wo=%v QNorm=%v KNorm=%v SSMInProj=%v",
		l3.Wq != nil, l3.Wk != nil, l3.Wv != nil, l3.Wo != nil,
		l3.AttnQNorm != nil, l3.AttnKNorm != nil, l3.SSMInProj != nil)
	if l3.Wq != nil {
		t.Logf("  Wq: rows=%d cols=%d", l3.Wq.Rows, l3.Wq.Cols)
	}

	t.Logf("SSMRun allocated: %v", pipe.RunState.SSMRun != nil)
	t.Logf("SSMState allocated: %v", pipe.RunState.SSMState != nil)

	cfg := DefaultGenerateConfig()
	cfg.MaxTokens = 32
	cfg.Sampler.Temperature = 0

	prompt := "The capital of France is"
	text, tokPerSec, err := pipe.GenerateText(prompt, cfg)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}
	t.Logf("Output: %q", text)
	t.Logf("Speed: %.1f tok/s", tokPerSec)
}
