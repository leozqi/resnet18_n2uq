# Debug: TLMAC vs N2UQ Accuracy Divergence

## Symptoms (from Colab run)

| Metric | N2UQ vs Baseline | TLMAC vs Baseline | TLMAC vs N2UQ |
|--------|-----------------|-------------------|---------------|
| Top-1 agreement | 0.0% | 0.0% | 0.0% |
| Mean L2 distance | 48.32 | 19860.10 | 19855.97 |
| Logit Pearson r | 0.5165 | 0.1276 | 0.2106 |

Per-layer conv comparison (same LTQ activations fed to both):

| Layer | Pearson r | Rel err | Max err |
|-------|-----------|---------|---------|
| layer1.0.conv1 | -0.0203 | 4.74 | 10.74 |
| layer4.1.conv1 | -0.3966 | 2.88 | 2.50 |
| **Mean** | **-0.1190** | — | — |

**Key observation**: Pearson r is near-zero/negative — the TLMAC conv
output is *uncorrelated* with N2UQ, even when fed identical activations.
The LUT INIT integrity check passes (hex decoding is correct), so the
bug is in the **emulation/routing logic**, not the LUT truth tables.

---

## Issue 1 (CRITICAL): Routing is step-dependent, not static

### Paper (§4)

> "Depending on the `step` input to the PE, the selection of the LUT
> array changes. A mapping associating each step with the corresponding
> selection input for each multiplexer is stored in a read-only memory
> block."

The paper describes **two ROMs**:
1. `cluster_map` ROM: step → `wg_sel` (selects weight group within LUT array)
2. **routing ROM**: step → MUX selection (selects which LUT array drives each output)

### What the compile exports

`routing_bitmap.hex` is a **static OR'd bitmap**: `R[arr, p] = 1` if
array `arr` *ever* drives output `p` for *any* cluster. This is the wire
count metric `R = Σ I(∃c: R(e,c,p)≠0)` from §5.2 — a physical routing
congestion metric, **not** a functional routing map.

### What the emulator does

```python
route_map[p] = first array with bit set in R[:, p]
```

Picks ONE static array per output. But each output is driven by
**different arrays at different steps** (the array depends on which
cluster the step belongs to and which weight group that output needs).

### Evidence

For `layer1_0_conv1` (D_p=192, n_arr=479):

| Arrays per output | Count |
|-------------------|-------|
| 0 | 0 |
| 1 | 0 |
| >1 (29–47) | 192 |

**Every output is driven by 29–47 arrays.** The emulator picks the
first — essentially random. This scrambles the MAC results, producing
uncorrelated output (negative Pearson r).

### Correct behavior

The array driving output `p` at step `s` is:
```
route_map[s, p] = best_placement[cluster_map[s]][weight_group[s, p]]
```

This is a 2D per-step routing map `[D_s, D_p]`, not a 1D static map.

### Root cause

- **Compile bug**: exports static OR'd bitmap instead of per-step routing
- **Emulator bug**: uses `route_map[p]` instead of `route_map[s, p]`

### Fix

Export `route_map[D_s, D_p]` from `tlmac_compile.ipynb` (the compile
has `best_pp`, `cluster_map`, and `wg` — all needed to compute it).
The emulator loads this per-step map and uses `route_map[s, p]`.

---

## Issue 2 (SIGNIFICANT): Missing weight dequantization bias

### The math

N2UQ weight quantization (from `resnet.py`):
```
qw = sf * (qi/n - clip/2)
   where n = (2^BW-1)/clip = 3.5, clip = 2.0, sf = gamma*mean(|w|)
```

TLMAC LUT stores signed `w_int = qi - WEIGHT_OFFSET = qi - 3`.
The MAC computes `psum = Σ a_g · w_int_g`.

Deriving the relationship:
```
qw = sf * (qi/n - clip/2)
   = sf/n * (w_int + 3) - sf*clip/2
   = sf/n * w_int + sf*(3/n - clip/2)
   = sf/n * w_int - sf/7          [3/3.5 - 1 = -1/7]
```

So:
```
y = conv(x, qw) = sf/n * conv(x, w_int) - sf/7 * conv(x, 1)
```

With `a = n_q * x` (activation quantization, `n_q = 3.5`):
```
psum = conv(a, w_int) = n_q * conv(x, w_int)
y = sf/(n·n_q) * psum - sf/7 * conv(x, 1)
```

### What the emulator does

```python
y = sf / (n * n_q) * psum    # MISSING the -sf/7 * conv(x, 1) bias
```

### Why WEIGHT_OFFSET=3 causes this

The N2UQ quantization center (`cw=0`) maps to `qi = round(3.5) = 4`,
not 3. The true signed center is `qi - 3.5`, but the LUT stores
`qi - 3` (integer). This 0.5 offset per weight produces a constant
bias of `-sf/7 · conv(x, 1)` in the output.

### Magnitude

- `sf ≈ 1.75 · 0.03 ≈ 0.05` (typical)
- `sf/7 ≈ 0.007`
- `conv(x, 1) ≈ 576` (sum of 64×9 activations ≈ 1.0)
- bias ≈ 4.0, output ≈ 14 → **~28% error**

This explains the magnitude drift (rel err 2–7×) but NOT the correlation
loss (that's Issue 1). After fixing routing, this bias must still be
corrected.

### Note

This bias is **fundamental** — it exists regardless of WEIGHT_OFFSET
choice. With WEIGHT_OFFSET=4, the bias is `+sf/7`. With unsigned `qi`,
the bias is `-sf*clip/2 = -sf`. The N2UQ quantization `qw = sf*(qi/n - clip/2)`
always has the `-clip/2` term.

### Fix

Add bias correction to `forward_scaled`:
```python
conv_x_ones = F.conv2d(x, torch.ones(D_o, D_i, 3, 3), stride, padding)
y = sf/(n*n_q) * psum - sf/7 * conv_x_ones
```

Or: store `w_int = 2*qi - 7` (= `2*(qi-3.5)`) in the LUT, then
`y = sf/(2*n*n_q) * psum` with no bias. NLUT=5 still fits (range [-6,8]).

---

## Issue 3 (MAJOR): SA placement allows group collisions

### The constraint (paper §5)

> 1. Each LUT array can hold max `N_clus` weight groups.
> 2. Access is exclusive (one group per select slot).
> 3. Select signal `s` is shared across all arrays.

A LUT array has `NCLUS=8` slots (one per cluster). For cluster `ci`,
each array stores **at most one** group at slot `ci`. Two groups in
the **same cluster** cannot share an array.

### What the SA does

```python
assign_c[ci] = torch.randint(0, n_arr, (n_g,))  # random, no collision check
```

Random assignment with `n_g ≈ 500` groups into `n_arr ≈ 500` arrays.
By the birthday paradox, **~63% collision rate** — many groups land on
the same array within the same cluster.

### What happens on collision

The compile builds `group_at[ai, ci] = gid` via scatter. If two groups
`g0, g1` in cluster `ci` both map to array `ai`, the scatter **overwrites**
— last write wins. `g0` is silently lost. The LUT stores `g1` at slot `ci`.

Any output needing `g0` at cluster `ci` gets `g1` instead → **wrong MAC**.

### The SA doesn't penalize collisions

The SA minimizes route count `R = Σ I(∃c: R(e,c,p)≠0)`. Two same-cluster
groups on the same array drive the same outputs (OR'd), so they don't
increase `R`. The SA has **no incentive** to spread groups across arrays.

### Fix

The SA must enforce per-cluster **injection**: no two groups of the same
cluster share an array. Options:
1. Penalize collisions in the energy function.
2. Constrain moves to maintain injection (swap only between non-colliding positions).
3. Use a bipartite matching / assignment algorithm per cluster.

---

## Issue 4 (MINOR): routing_bitmap.hex format is unusable

The exported `routing_bitmap.hex` (static OR'd bitmap) is a wire-count
metric for FPGA physical routing, not a functional routing map. It
cannot be used by the emulator to determine which array drives each
output at each step.

Every output has 29–47 arrays set — picking "first" is meaningless.

### Fix

Replace `routing_bitmap.hex` with `route_map.hex`: a per-step routing
map `[D_s, D_p]` of array indices. Each entry is the array index (0–511)
that drives that output at that step. This matches the paper's
"per-step routing ROM" (§4).

---

## Severity & Fix Order

| # | Issue | Severity | Effect | Fix Location |
|---|-------|----------|--------|--------------|
| 1 | Static routing | CRITICAL | Uncorrelated output (r≈0) | compile + emulator |
| 3 | SA collisions | MAJOR | Lost weight groups | compile (SA) |
| 2 | Missing bias | SIGNIFICANT | ~28% magnitude error | emulator |
| 4 | routing_bitmap format | MINOR | Root cause of #1 | compile |

**Fix order**: Issue 3 (SA) → Issue 1/4 (routing export + emulator) →
Issue 2 (bias). Issue 3 must be fixed first because collisions corrupt
the placement, making the per-step routing map wrong even if exported
correctly.

## What is CORRECT (verified)

- **LUT INIT integrity**: hex decoding, bit-packing, 2's-complement sign
  extension — all verified by the integrity check (ALL PASS).
- **D_p/D_s reshape**: `D_p = 64·Dk = 192`, `D_s = D_i·D_o/64`, weight
  group = kernel row. Matches paper §3.2.
- **LUT address**: `{wg_sel[2:0], act_bits[2:0]}` = 6 bits. Matches paper.
- **Activation bit packing**: bit `g` of `act_val` = column `g`
  activation, multiplied by weight `g`. Matches compile.
- **Bit-serial loop**: LSB-first, shift-left by `b`, accumulate. Matches
  paper §3.1.2, §4.
- **Activation quantization**: LTQ output [0,2] → [0,7] via
  `round(x·3.5)`. Correct (LTQ produces 8 quantized levels).
- **NLUT=5**: covers MAC range [-9, 12]. Correct.
- **NCLUS=8, G=3, BW=3, BA=3**: all match paper.
- **Spectral clustering + Cluster QR**: matches paper §5.1.
- **SA algorithm structure**: matches paper §5.2 Algorithm 1 (but
  missing collision constraint — Issue 3).
