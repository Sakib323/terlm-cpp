#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

MAGIC = b"THMDF32\x00"
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


def rmsnorm(input_tensor, weight, epsilon):
    inverse_rms = torch.rsqrt(
        input_tensor.square().mean(dim=-1, keepdim=True) + epsilon
    )
    return input_tensor * inverse_rms * weight


def causal_depthwise_conv_silu(input_tensor, weight, bias, initial_cache):
    batch, sequence, _ = input_tensor.shape
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

    return output.transpose(1, 2).reshape(batch, sequence, hidden), state


def attention_block(input_tensor, weights, initial_conv, initial_recurrent,
                    lower_bound, epsilon, apply_lower_bound):
    convolved, final_conv = causal_depthwise_conv_silu(
        input_tensor,
        weights["conv_weight"],
        weights["conv_bias"],
        initial_conv,
    )

    projected_i = fused_bitlinear(
        convolved,
        weights["i_proj_norm_weight"],
        weights["i_proj_weight"],
    )

    projected_f = torch.sigmoid(fused_bitlinear(
        convolved,
        weights["f_proj_norm_weight"],
        weights["f_proj_weight"],
    ))

    if apply_lower_bound:
        projected_f = lower_bound + (1.0 - lower_bound) * projected_f

    projected_g = fused_bitlinear(
        convolved,
        weights["g_proj_norm_weight"],
        weights["g_proj_weight"],
    )

    content = F.silu(projected_i) * (1.0 - projected_f)
    recurrent, final_recurrent = hgrn_recurrent(
        content,
        projected_f,
        initial_recurrent,
    )

    gated = rmsnorm(recurrent, weights["gnorm_weight"], epsilon)
    gated = gated * F.silu(projected_g)

    output = fused_bitlinear(
        gated,
        weights["o_proj_norm_weight"],
        weights["o_proj_weight"],
    )

    return output, final_conv, final_recurrent


def decoder_layer(input_tensor, weights, initial_conv, initial_recurrent,
                  lower_bound, epsilon, apply_lower_bound):
    attention, final_conv, final_recurrent = attention_block(
        input_tensor,
        weights,
        initial_conv,
        initial_recurrent,
        lower_bound,
        epsilon,
        apply_lower_bound,
    )

    after_attention = input_tensor + attention
    gate_and_value = fused_bitlinear(
        after_attention,
        weights["gate_proj_norm_weight"],
        weights["gate_proj_weight"],
    )
    gate, value = gate_and_value.chunk(2, dim=-1)
    mlp_hidden = F.silu(gate) * value

    mlp = fused_bitlinear(
        mlp_hidden,
        weights["down_proj_norm_weight"],
        weights["down_proj_weight"],
    )

    return after_attention + mlp, final_conv, final_recurrent


def model_forward(token_ids, embedding_weight, layers, raw_lower_bounds,
                  final_norm_weight, lm_head_norm_weight, lm_head_weight,
                  initial_conv_state, initial_recurrent_state, epsilon):
    hidden_states = embedding_weight[token_ids]

    lower_bounds = raw_lower_bounds.softmax(dim=0)
    lower_bounds = lower_bounds.cumsum(dim=0) - lower_bounds.cumsum(dim=0)[0]

    final_conv = []
    final_recurrent = []

    for layer_index, layer_weights in enumerate(layers):
        hidden_states, conv, recurrent = decoder_layer(
            hidden_states,
            layer_weights,
            initial_conv_state[layer_index],
            initial_recurrent_state[layer_index],
            lower_bounds[layer_index],
            epsilon,
            apply_lower_bound=layer_index > 0,
        )
        final_conv.append(conv)
        final_recurrent.append(recurrent)

    hidden_states = rmsnorm(hidden_states, final_norm_weight, epsilon)
    logits = fused_bitlinear(
        hidden_states,
        lm_head_norm_weight,
        lm_head_weight,
    )

    return logits, torch.stack(final_conv), torch.stack(final_recurrent)


def make_layer(hidden, intermediate, kernel_size, values):
    return {
        "conv_weight": values(hidden, kernel_size, scale=0.20),
        "conv_bias": values(hidden, scale=0.10),
        "i_proj_norm_weight": 0.9 + values(hidden, scale=0.15),
        "i_proj_weight": values(hidden, hidden, scale=0.15),
        "f_proj_norm_weight": 0.9 + values(hidden, scale=0.15),
        "f_proj_weight": values(hidden, hidden, scale=0.15),
        "g_proj_norm_weight": 0.9 + values(hidden, scale=0.15),
        "g_proj_weight": values(hidden, hidden, scale=0.15),
        "gnorm_weight": 0.9 + values(hidden, scale=0.15),
        "o_proj_norm_weight": 0.9 + values(hidden, scale=0.15),
        "o_proj_weight": values(hidden, hidden, scale=0.15),
        "gate_proj_norm_weight": 0.9 + values(hidden, scale=0.15),
        "gate_proj_weight": values(2 * intermediate, hidden, scale=0.15),
        "down_proj_norm_weight": 0.9 + values(intermediate, scale=0.15),
        "down_proj_weight": values(hidden, intermediate, scale=0.15),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="tests/fixtures/hgrn-model-parity-f32.bin",
    )
    args = parser.parse_args()

    torch.manual_seed(20260717)
    torch.set_num_threads(1)

    batch = 1
    sequence = 5
    vocab = 17
    layers_count = 2
    hidden = 12
    heads = 3
    intermediate = 20
    kernel_size = 3
    epsilon = 1e-5
    head_dim = hidden // heads

    def values(*shape, scale):
        return torch.randn(*shape, dtype=torch.float32) * scale

    token_ids = torch.tensor([[3, 1, 14, 5, 9]], dtype=torch.uint32)
    embedding_weight = values(vocab, hidden, scale=0.7)
    raw_lower_bounds = values(layers_count, hidden, scale=0.4)
    final_norm_weight = 0.9 + values(hidden, scale=0.15)
    lm_head_norm_weight = 0.9 + values(hidden, scale=0.15)
    lm_head_weight = values(vocab, hidden, scale=0.15)

    layers = [
        make_layer(hidden, intermediate, kernel_size, values)
        for _ in range(layers_count)
    ]

    initial_conv_state = values(
        layers_count,
        batch,
        hidden,
        kernel_size,
        scale=0.4,
    )
    initial_recurrent_state = values(
        layers_count,
        batch,
        heads,
        head_dim,
        scale=0.4,
    )

    logits, final_conv_state, final_recurrent_state = model_forward(
        token_ids.to(torch.long),
        embedding_weight,
        layers,
        raw_lower_bounds,
        final_norm_weight,
        lm_head_norm_weight,
        lm_head_weight,
        initial_conv_state,
        initial_recurrent_state,
        epsilon,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layer_tensor_names = (
        "conv_weight",
        "conv_bias",
        "i_proj_norm_weight",
        "i_proj_weight",
        "f_proj_norm_weight",
        "f_proj_weight",
        "g_proj_norm_weight",
        "g_proj_weight",
        "gnorm_weight",
        "o_proj_norm_weight",
        "o_proj_weight",
        "gate_proj_norm_weight",
        "gate_proj_weight",
        "down_proj_norm_weight",
        "down_proj_weight",
    )

    with output_path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack(
            "<9If",
            VERSION,
            batch,
            sequence,
            vocab,
            layers_count,
            hidden,
            heads,
            intermediate,
            kernel_size,
            epsilon,
        ))
        handle.write(token_ids.contiguous().numpy().tobytes())

        for tensor in (
            embedding_weight,
            raw_lower_bounds,
            final_norm_weight,
            lm_head_norm_weight,
            lm_head_weight,
        ):
            write_tensor(handle, tensor)

        for layer_weights in layers:
            for name in layer_tensor_names:
                write_tensor(handle, layer_weights[name])

        for tensor in (
            initial_conv_state,
            initial_recurrent_state,
            logits,
            final_conv_state,
            final_recurrent_state,
        ):
            write_tensor(handle, tensor)

    print(f"Wrote {output_path}")
    print(
        f"shape: batch={batch}, sequence={sequence}, vocab={vocab}, "
        f"layers={layers_count}, hidden={hidden}, heads={heads}, "
        f"intermediate={intermediate}, kernel={kernel_size}"
    )


if __name__ == "__main__":
    main()
