#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

MAGIC = b"TFBLF32\x00"
VERSION = 1


def rmsnorm(x, weight, epsilon):
    inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon)
    return x * inverse_rms * weight


def activation_quant(x):
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    return (x * scale).round().clamp(-128, 127) / scale


def weight_quant(weight):
    scale = 1.0 / weight.abs().mean().clamp_min(1e-5)
    return (weight * scale).round().clamp(-1, 1) / scale


def fused_bitlinear(input_tensor, norm_weight, linear_weight, epsilon):
    normalized = rmsnorm(input_tensor, norm_weight, epsilon)
    quantized_input = activation_quant(normalized)
    quantized_weight = weight_quant(linear_weight)
    return F.linear(quantized_input, quantized_weight)


def write_tensor(handle, tensor):
    value = tensor.detach().cpu().contiguous().to(torch.float32)
    handle.write(value.numpy().tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="tests/fixtures/fused-bitlinear-parity-f32.bin",
    )
    args = parser.parse_args()

    torch.manual_seed(20260717)
    torch.set_num_threads(1)

    rows = 5
    input_features = 12
    output_features = 9
    epsilon = 1e-6

    input_tensor = torch.randn(rows, input_features, dtype=torch.float32) * 0.8
    norm_weight = 0.75 + torch.randn(input_features, dtype=torch.float32) * 0.2
    linear_weight = torch.randn(
        output_features,
        input_features,
        dtype=torch.float32,
    ) * 0.25

    expected_output = fused_bitlinear(
        input_tensor,
        norm_weight,
        linear_weight,
        epsilon,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(
            struct.pack(
                "<4If",
                VERSION,
                rows,
                input_features,
                output_features,
                epsilon,
            )
        )
        write_tensor(handle, input_tensor)
        write_tensor(handle, norm_weight)
        write_tensor(handle, linear_weight)
        write_tensor(handle, expected_output)

    print(f"Wrote {output_path}")
    print(
        f"shape: rows={rows}, in={input_features}, "
        f"out={output_features}, epsilon={epsilon}"
    )


if __name__ == "__main__":
    main()
