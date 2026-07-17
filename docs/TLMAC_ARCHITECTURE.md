# TLMAC Architecture Overview

Summary of the TLMAC PE architecture from Gerlinghoff et al., FPGA 2024.
Relevant code: `tlmac_compile.ipynb` (weight compiler), `test_accuracy.ipynb` (emulator).

## Core Concept

TLMAC replaces multiply-accumulate (MAC) operations with LUT-6 lookup tables,
exploiting the low-bit quantization of weights. Each LUT array stores up to
`N_clus` weight groups, selected per-step via a 3-bit select signal. The MAC
is computed bit-serially over `B_a` iterations (one bit of each activation
at a time).

## Key Parameters (3-bit weights, 3-bit activations)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `G`       | 3     | Weight group size (= kernel row size `D_k`) |
| `B_w`     | 3     | Weight bit width |
| `B_a`     | 3     | Activation bit width (bit-serial iterations) |
| `N_clus`  | 8     | Select codes per LUT array (`2^(6-G)`) |
| `N_lut`   | 5     | LUT-6 per array (`B_w + ceil(log2(G))`) |
| `D_p`     | 192   | Parallel outputs per PE (`64 * D_k`) |
| `D_s`     | varies| Sequential steps (`D_i * D_o / 64`) |

## Hardware Components (paper §4)

```
┌─────────────────────────────────────────────────────────────┐
│ TLMAC Processing Element                                     │
│                                                              │
│  ┌──────────────┐    ┌───────────────────┐    ┌──────────┐  │
│  │ Cluster Map  │    │    LUT Pool       │    │ Switch   │  │
│  │ ROM          │    │                   │    │ Network  │  │
│  │ step → wg_sel│    │ ┌───┐┌───┐ ┌───┐ │    │ (MUXes)  │  │
│  └──────┬───────┘    │ │LUT││LUT│ │LUT│ │    │          │  │
│         │            │ │arr││arr│ │arr│ │    │ ┌─MUX──┐ │  │
│    step input        │ │ 0 ││ 1 │ │N │ │────│→│output │ │  │
│         │            │ └───┘└───┘ └───┘ │    │ │sel[0] │ │  │
│         └───────────→│  ↑ shared        │    │ └───────┘ │  │
│                      │  act bits        │    │    ...     │  │
│  ┌──────────────┐    └───────────────────┘    │ ┌─MUX──┐ │  │
│  │ Routing ROM  │                             │→│output │ │  │
│  │ step → mux   │←────────────────────────────│ │sel[Dp│ │  │
│  └──────────────┘                             │ └───────┘ │  │
│                                               └─────┬─────┘  │
│                                                     │        │
│                                               ┌─────▼─────┐  │
│                                               │Accumulators│  │
│                                               │ (D_p regs) │  │
│                                               └───────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Cluster Map ROM
- Maps `step` → `wg_sel` (3-bit select)
- Selects which weight group within each LUT array to use
- Contents: decided during clustering (paper §5.1)

### Routing ROM (per-step)
- Maps `step` → MUX selection for each of the `D_p` outputs
- Selects which LUT array drives each output at each step
- Contents: derived from SA placement (paper §5.2)
- **This is step-dependent** — different steps may route different arrays

### LUT Pool
- `N_arr` LUT arrays, each storing `N_clus=8` weight groups
- Each array: `N_lut=5` LUT-6, producing 5-bit signed MAC result
- Address = `{wg_sel[2:0], act_bits[2:0]}` = 6 bits (64 entries)
- `N_arr` determined by place & route (max unique groups per cluster)

### Switch Network
- `D_p` MUXes, each connected to a subset of LUT arrays
- Static wire connections (minimized by SA), dynamic MUX select (from ROM)

### Accumulators
- `D_p` registers of `B_p` bits each
- Loaded from partial sum buffer before processing
- Accumulated over `B_a` bit-serial iterations (LSB first, shift-left by `b`)
- Output after all `B_a` bits processed

## Bit-Serial MAC (paper §3.1.2)

```
p = Σ_{b=0}^{B_a-1} 2^b (a_0^b · w_0 + a_1^b · w_1 + ... + a_{G-1}^b · w_{G-1})
```

- `a_g^b` = bit `b` of the `B_a`-bit activation `a_g`
- `w_g` = signed weight from weight group (stored in LUT)
- Processing: LSB first, each iteration shifts result left by `b`

## Weight Tensor Layout (paper §3.2)

```
[D_o, D_i, D_k, D_k]  →  [D_s, D_p, D_k]

D_p = 64 * D_k = 192    (parallel outputs: 64 channels × 3 kernel rows)
D_s = D_i * D_o / 64    (sequential steps: input channels × output blocks)
G = D_k = 3             (weight group = one kernel row = 3 column weights)
```

Output channel `oc` and kernel row `kr` map to parallel index:
`p = oc * D_k + kr`

## Place & Route (paper §5)

### Step 1: Spectral Clustering (§5.1)
- Extract unique weight groups, build binary assignment matrix `C[D_s, N_uwg]`
- Spectral clustering + Cluster QR on `C` → `N_clus=8` clusters
- Each cluster = set of steps sharing similar weight groups
- Steps in same cluster share the same `wg_sel` select code

### Step 2: SA Routing Reduction (§5.2)
- Assign each cluster's weight groups to `N_arr` LUT arrays
- Minimize wire count `R = Σ_{e,p} I(∃c: R(e,c,p)≠0)`
- Swap move: exchange groups between arrays within same cluster
- Constraint: injection per cluster (no two groups in same cluster on same array)

## N2UQ Integration

The TLMAC PE replaces only the 3×3 convolution layers. All other layers
(LTQ quantization, learnable bias, PReLU, batch normalization, skip
connections) remain in floating-point, computed by DSP slices or host.

### Dequantization (outside PE)

```
qw = sf * (qi/n - clip/2)
   = sf/n * (qi - WEIGHT_OFFSET) - sf*(clip/2 - WEIGHT_OFFSET/n)
```

The PE outputs integer `psum = conv(a, w_int)` where `w_int = qi - WEIGHT_OFFSET`.
The floating-point layer applies:

```
y = sf/(n·n_q) · psum + sf·(WEIGHT_OFFSET/n - clip/2) · conv(x, ones)
```

where `n = (2^BW-1)/clip`, `n_q = (2^BA-1)/2`.

## References

- Gerlinghoff et al., "TLMAC: Table Lookup Multiply-Accumulate for Quantised
  Neural Networks on FPGAs", FCCM 2024.
- Paper text: `/context/team-05-tlmac/docs/01_paper.md`
- RTL: `/context/team-05-tlmac/rtl/` (TODO stubs)
