# TerLM HGRN CPU Kernel

## Scope

This directory initially provides a scalar float32 reference for the
forward-only HGRN recurrent update used during TerLM inference.

## Tensor layout

`x` and `g` use contiguous `[batch, heads, sequence, head_dim]` layout.

`initial_state` and `final_state` use contiguous `[batch, heads, head_dim]`
layout. Passing a null `initial_state` represents an all-zero initial state.

## Recurrence

For each batch `b`, head `h`, token `t`, and feature `d`:

```text
state[b,h,d] = g[b,h,t,d] * state[b,h,d] + x[b,h,t,d]
output[b,h,t,d] = state[b,h,d]
```

## Numerical contract

The scalar float32 implementation is the reference. Future AVX2, AVX-512,
and NEON paths must match it within an absolute error of 1e-5.
