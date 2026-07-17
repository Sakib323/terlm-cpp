#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

MAGIC = b"THMLPF32"
VERSION = 1
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


def attention_mlp(
    input_tensor,
    gate_proj_norm_weight,
    gate_proj_weight,
    down_proj_norm_weight,
    down_proj_weight,
):
    gate_and_value = fused_bitlinear(
        input_tensor,
        gate_proj_norm_weight,
        gate_proj_weight,
    )

    gate, value = gate_and_value.chunk(2, dim=-1)
    swiglu_output = F.silu(gate) * value

    return fused_bitlinear(
        swiglu_output,
        down_proj_norm_weight,
        down_proj_weight,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="tests/fixtures/hgrn-mlp-parity-f32.bin",
    )
    args = parser.parse_args()

    torch.manual_seed(20260717)
    torch.set_num_threads(1)

    batch = 2
    sequence = 5
    hidden = 12
    intermediate = 20

    def values(*shape, scale):
        return torch.randn(*shape, dtype=torch.float32) * scale

    input_tensor = values(batch, sequence, hidden, scale=0.7)

    gate_proj_norm_weight = 0.8 + values(hidden, scale=0.15)
    gate_proj_weight = values(2 * intermediate, hidden, scale=0.15)

    down_proj_norm_weight = 0.8 + values(intermediate, scale=0.15)
    down_proj_weight = values(hidden, intermediate, scale=0.15)

    output = attention_mlp(
        input_tensor,
        gate_proj_norm_weight,
        gate_proj_weight,
        down_proj_norm_weight,
        down_proj_weight,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(
            struct.pack(
                "<4I",
                VERSION,
                batch,
                sequence,
                hidden,
            )
        )
        handle.write(struct.pack("<I", intermediate))

        for tensor in (
            input_tensor,
            gate_proj_norm_weight,
            gate_proj_weight,
            down_proj_norm_weight,
            down_proj_weight,
            output,
        ):
            write_tensor(handle, tensor)

    print(f"Wrote {output_path}")
    print(
        f"shape: batch={batch}, sequence={sequence}, "
        f"hidden={hidden}, intermediate={intermediate}"
    )


if __name__ == "__main__":
    main()
