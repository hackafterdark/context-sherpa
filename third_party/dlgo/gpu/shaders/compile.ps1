$shaders = @(
    @{name="matvec_f32"; file="matvec_f32.comp"},
    @{name="matvec_q4_0"; file="matvec_q4_0.comp"},
    @{name="matvec_q8_0"; file="matvec_q8_0.comp"},
    @{name="matvec_q4_k"; file="matvec_q4_k.comp"},
    @{name="matvec_q5_0"; file="matvec_q5_0.comp"},
    @{name="matvec_q6_k"; file="matvec_q6_k.comp"},
    @{name="attention"; file="attention.comp"},
    @{name="rmsnorm"; file="rmsnorm.comp"},
    @{name="rmsnorm_heads"; file="rmsnorm_heads.comp"},
    @{name="softmax"; file="softmax.comp"},
    @{name="rope"; file="rope.comp"},
    @{name="swiglu"; file="swiglu.comp"},
    @{name="geglu"; file="geglu.comp"},
    @{name="gelu"; file="gelu.comp"},
    @{name="add"; file="add.comp"},
    @{name="scale"; file="scale.comp"},
    @{name="add_rmsnorm"; file="add_rmsnorm.comp"}
)

$header = @"
// Auto-generated SPIR-V shader data. Do not edit.
#ifndef DLGO_SHADERS_SPIRV_H
#define DLGO_SHADERS_SPIRV_H

#include <stdint.h>
#include <stddef.h>

"@

foreach ($s in $shaders) {
    $spvFile = "$($s.name).spv"
    Write-Host "Compiling $($s.file) -> $spvFile"
    & glslc --target-env=vulkan1.2 -O $s.file -o $spvFile
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to compile $($s.file)"
        exit 1
    }

    $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $spvFile))
    $header += "static const uint32_t spv_$($s.name)[] = {`n"

    for ($i = 0; $i -lt $bytes.Length; $i += 4) {
        $val = [BitConverter]::ToUInt32($bytes, $i)
        $header += "    0x$($val.ToString('x8')),"
        if (($i / 4 + 1) % 8 -eq 0) { $header += "`n" } else { $header += " " }
    }
    $header += "`n};`n"
    $header += "static const size_t spv_$($s.name)_size = sizeof(spv_$($s.name));`n`n"
}

$header += @"

typedef struct {
    const char* name;
    const uint32_t* code;
    size_t code_size;
    int num_buffers;
    int push_const_size;
} ShaderInfo;

static const ShaderInfo shader_registry[] = {
    {"matvec_f32",  spv_matvec_f32,  spv_matvec_f32_size,  3, 8},   // PIPE_MATVEC_F32
    {"matvec_f32",  spv_matvec_f32,  spv_matvec_f32_size,  3, 8},   // PIPE_MATVEC_F16 (placeholder)
    {"matvec_q4_0", spv_matvec_q4_0, spv_matvec_q4_0_size, 3, 8},   // PIPE_MATVEC_Q4_0
    {"matvec_q8_0", spv_matvec_q8_0, spv_matvec_q8_0_size, 3, 8},   // PIPE_MATVEC_Q8_0
    {"matvec_q4_k", spv_matvec_q4_k, spv_matvec_q4_k_size, 3, 8},   // PIPE_MATVEC_Q4_K
    {"matvec_q5_0", spv_matvec_q5_0, spv_matvec_q5_0_size, 3, 8},   // PIPE_MATVEC_Q5_0
    {"matvec_q6_k", spv_matvec_q6_k, spv_matvec_q6_k_size, 3, 8},   // PIPE_MATVEC_Q6_K
    {"matvec_q4_0", spv_matvec_q4_0, spv_matvec_q4_0_size, 3, 8},   // PIPE_DEQUANT_Q4_0 (placeholder)
    {"matvec_q8_0", spv_matvec_q8_0, spv_matvec_q8_0_size, 3, 8},   // PIPE_DEQUANT_Q8_0 (placeholder)
    {"matvec_q4_0", spv_matvec_q4_0, spv_matvec_q4_0_size, 3, 8},   // PIPE_DEQUANT_Q4_K (placeholder)
    {"matvec_q4_0", spv_matvec_q4_0, spv_matvec_q4_0_size, 3, 8},   // PIPE_DEQUANT_Q5_0 (placeholder)
    {"matvec_q4_0", spv_matvec_q4_0, spv_matvec_q4_0_size, 3, 8},   // PIPE_DEQUANT_Q6_K (placeholder)
    {"rmsnorm",     spv_rmsnorm,     spv_rmsnorm_size,     3, 8},   // PIPE_RMSNORM
    {"softmax",     spv_softmax,     spv_softmax_size,     1, 4},   // PIPE_SOFTMAX
    {"rope",        spv_rope,        spv_rope_size,        2, 24},  // PIPE_ROPE
    {"swiglu",      spv_swiglu,      spv_swiglu_size,      3, 4},   // PIPE_SWIGLU
    {"geglu",       spv_geglu,       spv_geglu_size,       3, 4},   // PIPE_GEGLU
    {"gelu",        spv_gelu,        spv_gelu_size,        1, 4},   // PIPE_GELU
    {"add",         spv_add,         spv_add_size,         3, 4},   // PIPE_ADD
    {"add",         spv_add,         spv_add_size,         3, 4},   // PIPE_ADD_SCALED (placeholder)
    {"scale",       spv_scale,       spv_scale_size,       1, 8},   // PIPE_SCALE
    {"scale",       spv_scale,       spv_scale_size,       1, 8},   // PIPE_MUL (placeholder)
    {"scale",       spv_scale,       spv_scale_size,       1, 8},   // PIPE_COPY_F32 (placeholder)
    {"attention",   spv_attention,   spv_attention_size,   4, 24},  // PIPE_ATTENTION
    {"rmsnorm_heads", spv_rmsnorm_heads, spv_rmsnorm_heads_size, 2, 8}, // PIPE_RMSNORM_HEADS
    {"add_rmsnorm", spv_add_rmsnorm, spv_add_rmsnorm_size, 5, 8}, // PIPE_ADD_RMSNORM
};

#endif // DLGO_SHADERS_SPIRV_H
"@

$header | Out-File -FilePath "..\csrc\shaders_spirv.h" -Encoding ascii
Write-Host "Generated csrc/shaders_spirv.h"
