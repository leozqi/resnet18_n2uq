# Refactor test_accuracy.ipynb — TLMAC vs N2UQ Accuracy Verification

## Goals

Refactor `test_accuracy.ipynb` so the **highest priority** is an accurate,
faithful evaluation of the TLMAC architecture and its compiled weights
versus the N2UQ quantized model.  Keep all existing tests but reorganize
them in priority order, fix correctness issues, and vectorize hot paths.

## Key Insight: TLMAC weights ≠ N2UQ weights

TLMAC compiled weights are derived from N2UQ but are NOT bit-identical
because of:
1. The weight-group clustering step (spectral + Cluster QR)
2. The routing SA step (placement optimisation)

So we verify **accuracy is same or better**, not bit-exact match.
However, the LUT truth tables MUST exactly reproduce the N2UQ signed
integer weights — this is the correctness floor.

## What Changes

### 1. Reorganize notebook sections by priority

**Section 1: Setup** (unchanged)
- Clone repo, install deps, extract tar.gz
- Imports, device

**Section 2: Model Definitions** (unchanged)
- N2UQ architecture (LearnableBias, LTQ, HardQuantizeConv, BasicBlockQ, N2UQ_ResNet18)
- Load N2UQ + baseline models

**Section 3: TLMAC Emulator** (refactored — highest priority)
- `TLMACConvEmulator` class: faithfully reproduces paper §3.1.2, §4
- Loads hex files (LUT INIT, cluster_map, routing_bitmap)
- Bit-serial MAC: `p = Σ_{b=0}^{Ba-1} 2^b (Σ_g a_g^b · w_g)`
- LUT-6 lookup, switch-network routing, partial-sum accumulation
- **Fix**: activation quantization must match LTQ output range [0, 2]
- **Fix**: scaling must be `y = sf/(n·n_q) * psum`
- **Vectorize**: precompute `routed_table[D_p, 64]`, batch all spatial positions

**Section 4: LUT Truth-Table Verification** (moved up — correctness floor)
- For each layer, verify LUT INIT values match N2UQ signed integer weights
- Check: for sampled (step, address) pairs, LUT output == expected MAC
- This is the ground-truth check that hex files are correct

**Section 5: Per-Layer Conv Comparison** (new — isolates TLMAC conv accuracy)
- For each 3x3 conv layer, feed the SAME LTQ-quantized activations
- Compare TLMAC emulator output vs N2UQ `F.conv2d(x, qw)` output
- Reports per-layer Pearson r and relative error
- This isolates the conv computation from the rest of the model

**Section 6: Full-Model Forward Pass** (refactored)
- `tlmac_resnet18()`: replaces all 16 conv layers with TLMAC emulator
- Non-conv layers (LTQ, bias, PReLU, BN, skip) use N2UQ parameters
- Compares TLMAC vs N2UQ vs baseline on synthetic inputs
- Reports top-1 agreement, L2 distance, Pearson r

**Section 7: Accuracy on Real Data** (if ImageNet available)
- Run all three models on a small ImageNet subset
- Report top-1/top-5 accuracy for each

**Section 8: Visualization** (kept from original)
- Confidence histograms, per-sample distance, max-logit scatter

**Section 9: Summary**
- Table comparing all three models

### 2. Correctness Fixes

1. **Activation quantization**: LTQ output is in [0, 2]. Map to [0, 2^BA-1]
   via `round(x * (2^BA-1)/2)`. Previous version clamped to [-1, 1].

2. **Scaling**: `y = sf/(n·n_q) * psum` where:
   - `n = (2^BW-1)/clip` (weight quantization factor)
   - `n_q = (2^BA-1)/2` (activation quantization factor)
   - `sf = gamma * mean(|w|)` per output channel
   - Previous version used `sf/n` which ignored activation quantization.

3. **Per-layer conv comparison**: feed the same LTQ output to both TLMAC
   and N2UQ conv. This isolates the TLMAC conv accuracy from the rest
   of the model. Previous full-conv comparison had near-zero correlation
   because it used different activation quantization.

### 3. Performance Vectorization

- **Spatial dimension**: batch all oH×oW positions (already done)
- **routed_table**: precompute `lut_table[route_map]` → [D_p, 64] (already done)
- **Bit-serial inner loop**: keep as Python loop (D_s × DK × BA iterations),
  but each iteration is a vectorized tensor op over all spatial positions

### 4. What's Deleted

- Old cell 13 (synthetic data comparison, BL vs N2UQ only) — replaced by
  Section 6 which includes TLMAC
- Old cell 14 (visualization, BL vs N2UQ only) — replaced by Section 8
  which includes TLMAC
- Duplicate/legacy code paths

## Migration Steps

1. ✅ SA injection: enforce per-cluster injection in init + swap move
2. ✅ Per-step routing: export route_map[D_s, D_p] from compile, load in emulator
3. ✅ Bias correction: add -sf·(WOFFSET/n - clip/2)·conv(x,ones) to forward_scaled
4. ❌ Re-run compile on Colab (re-generates hex files with new routing)
5. ❌ Re-run test_accuracy on Colab (verifies accuracy fix)

## Open Questions

- Should we add ImageNet evaluation? (Needs ImageNet data on Colab)
- Should the per-layer comparison use random input or real LTQ output?
  → Real LTQ output, to match actual inference conditions
- **TLMAC paper description**: see `docs/TLMAC_ARCHITECTURE.md`
