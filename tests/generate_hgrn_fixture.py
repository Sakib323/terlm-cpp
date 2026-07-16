#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path

import torch


MAGIC = b"HGRNFX01"
VERSION = 1


def write_tensor(handle, tensor):
    tensor = tensor.detach().cpu().contiguous().to(torch.float32)
    handle.write(tensor.numpy().tobytes(order="C"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="tests/fixtures/hgrn-parity-f32.bin",
    )
    args = parser.parse_args()

    torch.manual_seed(20260716)

    batch = 2
    heads = 3
    sequence = 7
    head_dim = 5

    x = torch.randn(batch, heads, sequence, head_dim, dtype=torch.float32)
    g = torch.sigmoid(
        torch.randn(batch, heads, sequence, head_dim, dtype=torch.float32)
    )
    initial_state = torch.randn(batch, heads, head_dim, dtype=torch.float32)

    state = initial_state.clone()
    output = torch.empty_like(x)

    for token_index in range(sequence):
        state = g[:, :, token_index] * state + x[:, :, token_index]
        output[:, :, token_index] = state

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(
            struct.pack(
                "<IIQQQQ",
                VERSION,
                0,
                batch,
                heads,
                sequence,
                head_dim,
            )
        )
        write_tensor(handle, x)
        write_tensor(handle, g)
        write_tensor(handle, initial_state)
        write_tensor(handle, output)
        write_tensor(handle, state)

    print(f"Wrote fixture: {output_path}")
    print(
        "Shape: "
        f"[batch={batch}, heads={heads}, sequence={sequence}, head_dim={head_dim}]"
    )


if __name__ == "__main__":
    main()
