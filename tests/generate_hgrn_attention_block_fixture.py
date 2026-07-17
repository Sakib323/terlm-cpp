#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

MAGIC = b"THABF32\x00"
VERSION = 2
BITLINEAR_EPSILON = 1e-6


def write_tensor(handle, tensor):
    value = tensor.detach().cpu().contiguous().to(torch.float32)
    handle.write(value.numpy().tobytes())


def activation_quant(x):
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    return (x * scale).round().clamp(-128, 127) / scale


def weight_quant(weight):
    scale = 1.0 / weight.abs().mean().clamp_min(1e-5)
    return (weight * scale).round().clamp(-1, 1) / scale


def fused_bitlinear(input_tensor, norm_weight, linear_weight):
    inverse_rms = torch.rsqrt(
        input_tensor.square().mean(dim=-1, keepdim=True) + BITLINEAR_EPSILON
    )
    normalized = input_tensor * inverse_rms * norm_weight
    return F.linear(activation_quant(normalized), weight_quant(linear_weight))


def causal_depthwise_conv_silu(input_tensor, weight, bias, initial_cache):
    batch, sequence, hidden = input_tensor.shape
    kernel_size = weight.shape[1]

    cache = initial_cache.clone()
    output = torch.empty_like(input_tensor)

    for time in range(sequence):
        window = torch.cat(
            (cache[:, :, 1:], input_tensor[:, time, :].unsqueeze(-1)),
            dim=-1,
        )
        output[:, time, :] = F.silu(
            (window * weight.unsqueeze(0)).sum(dim=-1) + bias
        )
        cache = window

    return output, cache


def hgrn_recurrent(content, decay, initial_state):
    batch, sequence, hidden = content.shape
    heads = initial_state.shape[1]
    head_dim = hidden // heads

    content = content.view(batch, sequence, heads, head_dim).transpose(1, 2)
    decay = decay.view(batch, sequence, heads, head_dim).transpose(1, 2)

    state = initial_state.clone()
    output = torch.empty_like(content)

    for time in range(sequence):
        state = decay[:, :, time, :] * state + content[:, :, time, :]
        output[:, :, time, :] = state

    output = output.transpose(1, 2).reshape(batch, sequence, hidden)
    return output, state


def attention_block(
    input_tensor,
    conv_weight,
    conv_bias,
    i_proj_norm_weight,
    i_proj_weight,
    f_proj_norm_weight,
    f_proj_weight,
    g_proj_norm_weight,
    g_proj_weight,
    gnorm_weight,
    o_proj_norm_weight,
    o_proj_weight,
    initial_conv_cache,
    initial_recurrent_state,
    epsilon,
):
    convolved, final_conv_cache = causal_depthwise_conv_silu(
        input_tensor,
        conv_weight,
        conv_bias,
        initial_conv_cache,
    )

    projected_i = fused_bitlinear(
        convolved,
        i_proj_norm_weight,
        i_proj_weight,
    )

    projected_f = torch.sigmoid(
        fused_bitlinear(
            convolved,
            f_proj_norm_weight,
            f_proj_weight,
        )
    )

    projected_g = fused_bitlinear(
        convolved,
        g_proj_norm_weight,
        g_proj_weight,
    )

    content = F.silu(projected_i) * (1.0 - projected_f)

    recurrent_output, final_recurrent_state = hgrn_recurrent(
        content,
        projected_f,
        initial_recurrent_state,
    )

    inverse_rms = torch.rsqrt(
        recurrent_output.square().mean(dim=-1, keepdim=True) + epsilon
    )

    gated = (
        recurrent_output
        * inverse_rms
        * gnorm_weight
        * F.silu(projected_g)
    )

    output = fused_bitlinear(
        gated,
        o_proj_norm_weight,
        o_proj_weight,
    )

    return output, final_conv_cache, final_recurrent_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="tests/fixtures/hgrn-attention-block-parity-f32.bin",
    )
    args = parser.parse_args()

    torch.manual_seed(20260717)
    torch.set_num_threads(1)

    batch = 2
    sequence = 5
    hidden = 12
    heads = 3
    kernel_size = 3
    epsilon = 1e-5
    head_dim = hidden // heads

    def values(*shape, scale):
        return torch.randn(*shape, dtype=torch.float32) * scale

    input_tensor = values(batch, sequence, hidden, scale=0.7)
    conv_weight = values(hidden, kernel_size, scale=0.2)
    conv_bias = values(hidden, scale=0.1)

    i_proj_norm_weight = 0.8 + values(hidden, scale=0.15)
    i_proj_weight = values(hidden, hidden, scale=0.15)

    f_proj_norm_weight = 0.8 + values(hidden, scale=0.15)
    f_proj_weight = values(hidden, hidden, scale=0.15)

    g_proj_norm_weight = 0.8 + values(hidden, scale=0.15)
    g_proj_weight = values(hidden, hidden, scale=0.15)

    gnorm_weight = 0.8 + values(hidden, scale=0.15)

    o_proj_norm_weight = 0.8 + values(hidden, scale=0.15)
    o_proj_weight = values(hidden, hidden, scale=0.15)

    initial_conv_cache = values(batch, hidden, kernel_size, scale=0.4)
    initial_recurrent_state = values(batch, heads, head_dim, scale=0.4)

    output, final_conv_cache, final_recurrent_state = attention_block(
        input_tensor,
        conv_weight,
        conv_bias,
        i_proj_norm_weight,
        i_proj_weight,
        f_proj_norm_weight,
        f_proj_weight,
        g_proj_norm_weight,
        g_proj_weight,
        gnorm_weight,
        o_proj_norm_weight,
        o_proj_weight,
        initial_conv_cache,
        initial_recurrent_state,
        epsilon,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(
            struct.pack(
                "<6If",
                VERSION,
                batch,
                sequence,
                hidden,
                heads,
                kernel_size,
                epsilon,
            )
        )

        for tensor in (
            input_tensor,
            conv_weight,
            conv_bias,
            i_proj_norm_weight,
            i_proj_weight,
            f_proj_norm_weight,
            f_proj_weight,
            g_proj_norm_weight,
            g_proj_weight,
            gnorm_weight,
            o_proj_norm_weight,
            o_proj_weight,
            initial_conv_cache,
            initial_recurrent_state,
            output,
            final_conv_cache,
            final_recurrent_state,
        ):
            write_tensor(handle, tensor)

    print(f"Wrote {output_path}")
    print(
        f"shape: batch={batch}, sequence={sequence}, hidden={hidden}, "
        f"heads={heads}, kernel={kernel_size}, epsilon={epsilon}"
    )


if __name__ == "__main__":
    main()
