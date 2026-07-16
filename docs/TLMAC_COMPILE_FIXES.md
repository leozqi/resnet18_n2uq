# TLMAC Compile Notebook — Correctness Fixes

## Goal
Fix the correctness bugs in `tlmac_compile.ipynb` identified in the review
against the TLMAC paper (Gerlinghoff et al., FPGA 2024, §3.2, §5.1, §5.2,
Algorithm 1) and the RTL in `/context/team-05-tlmac/rtl/`. Keep all hot
paths vectorised in PyTorch so layer4 (D_i=256, D_o=512) stays fast on GPU.

## Paper reference (key equations)
- §3.2: `D_p = 64 * D_k = 192`, `D_s = D_i * D_o / 64`.
  Weight tensor reshaped `[D_o, D_i, D_k, D_k] -> [D_s, D_p, D_k]` where the
  `D_k` axis is the weight group (G = D_k = 3).
- §5.1: Assignment matrix `C ∈ B^{D_s x N_uwg}`. Cluster steps into exactly
  `N_clus = 2^(6-G) = 8` clusters. Paper uses **spectral clustering** +
  Cluster QR. `N_arr = max over clusters of |unique groups in cluster|`.
- §5.2 Algorithm 1: SA over placement. Routing matrix
  `R ∈ B^{N_arr x N_clus x D_p}`. Route count
  `R = Σ_e Σ_p 1[∃c: R(e,c,p) != 0]` — **per cluster**. Swap = move groups
  of cluster `c` between arrays `e_0, e_1`. Accept if
  `R_new < R_best or rand < exp((R_best - R_new - 1)/T)`, `T = I/(i+1)^1.4`.
- §3.1.2: One LUT array stores `N_clus` groups selected by `wg_sel`.
  INIT address = `{wg_sel[G-1:0] (actually [6-G-1:0]), act_bits[G-1:0]}`.
  `N_lut = B_w + ceil(log2(G))` output bits per array. Each array has ONE
  INIT set (64-bit per output bit), encoding all `N_clus` groups.

## RTL reference
- `lut_array.sv`: `INIT_FILE` (one file per array), `WEIGHT_GROUPS_PER_ARRAY`
  = N_clus, `SERIAL_BITS_PER_CYCLE` = N_lut, address = wg_sel + act_bits.
- `cluster_map_rom.sv`: `INIT_FILE = "cluster_map.hex"`, maps `step -> wg_sel`
  (3-bit for N_clus=8). One entry per step.
- `switch_network.sv`: `ROUTING_BITMAP [OUT_CHANNELS*ARR-1:0]` — flat 2D
  bitmap, row-major `arr * D_p + p` or `p * N_arr + arr` (decide and document).
- `lut_pool.sv`: `ARR` arrays, shared `act_bits_i` and `wg_sel_i`.

## Changes

### Cell "TLMAC Parameters & Helpers" — rewrite helpers

**`extract_weight_groups(w_q)`** — fix D_s/D_p (Bug 1, 2).
```
D_o, D_i, Dk, _ = w_q.shape          # [D_o, D_i, 3, 3]
D_p = 64 * Dk                        # 192 parallel outputs
D_s = D_i * D_o // 64                # sequential steps
# Reshape: [D_o, D_i, Dk, Dk] -> [D_s, D_p, Dk] weight groups.
# Each weight group = one kernel row of Dk weights.
# Output channels split into blocks of 64; each block x D_i input chs
# x Dk rows = 64*Dk = D_p groups per step.
w = w_q.permute(1, 0, 2, 3)          # [D_i, D_o, Dk, Dk]
# group kernel rows: [D_i, D_o, Dk, Dk] -> [D_i, D_o, Dk, Dk]
# We want [D_s, D_p, G] where G=Dk (one row).
# D_s = D_i * (D_o/64); D_p = 64*Dk.
o_blocks = D_o // 64
w = w.reshape(D_i, o_blocks, 64, Dk, Dk)     # [D_i, o_blk, 64, Dk, Dk]
# A step = (input_channel_block?) No: D_s = D_i * o_blocks.
# step index s iterates D_i then o_blocks: [D_i * o_blocks, 64*Dk, Dk]
w = w.reshape(D_i * o_blocks, 64 * Dk, Dk)   # [D_s, D_p, G]
w = w.reshape(-1, G)                          # [D_s*D_p, G]
unique, inverse = torch.unique(w, dim=0, return_inverse=True)
wg = inverse.reshape(D_s, D_p)
```
- Note: if `D_o % 64 != 0` (doesn't happen for ResNet-18: 64/128/256/512),
  pad to next multiple of 64 and mask extra outputs in routing.
- Return `wg` as a **torch int64 tensor on device** (not numpy) so downstream
  stays on GPU.

**`greedy_cluster_torch` -> `spectral_cluster_torch`** — fix Bug 3.
- Replace agglomerative merge with spectral clustering per paper.
- Build affinity matrix `A = C @ C.T` (cosine-ish: shared groups). For D_s up
  to ~2048 this is a 2048x2048 float32 = 16MB, fine on GPU.
- Symmetric normalised Laplacian `L = I - D^{-1/2} A D^{-1/2}`, take the
  `N_clus` smallest eigenvectors via `torch.linalg.eigh` (symmetric, fast).
  - For very large D_s, use `torch.linalg.eigh` on the `(D_s, D_s)` matrix;
    `eigh` on GPU handles 2048 trivially. If D_s > 4096, fall back to
    truncated eigendecomposition via `torch.linalg.svd_lowrank` or random
    projection (document).
- Cluster QR label assignment on the eigenvector embedding `U [D_s, N_clus]`:
  - Compute QR of `U.T` (`[N_clus, D_s]`), the argmax of abs(R) column norms
    gives cluster representatives; assign each point to the representative
    with max inner product. Vectorised:
    `Q, R = torch.linalg.qr(U_embed.T)`, pick `N_clus` pivots, assign by
    `U_embed @ Q` argmax. This is the Cluster QR algorithm.
- Fallback: if `torch.linalg.eigh` fails or D_s tiny, use
  `sklearn.cluster.SpectralClustering(n_clusters=N_clus)` (CPU).
- Returns `(clusters: list[set[int]], cg: list[set[int]])` as before.
- Keep the C-matrix construction vectorised (scatter, as now).

**`routing_sa`** — fix Bug 4, 5, 6.
- Input: `clusters`, `wg [D_s, D_p]` tensor, `n_arr`, `dp`.
- Build **per-cluster** group→outputs:
  `cg_outs[ci] = {gid: set(outputs)}` computed by scattering.
  Vectorised: for each cluster, gather its steps' rows of `wg`, then for
  each unique gid in the cluster, collect output indices. Use
  `torch.unique(wg[steps], return_inverse)` + scatter to build a
  `[n_groups_in_cluster, dp]` bool tensor of outputs per group. Store as
  list of `(gid_list, outs_bool [n_g, dp])` per cluster (on GPU).
- Placement state: `place[ci]` = `{gid: arr_idx}`. Represent as a tensor
  `assign_t [N_total_groups, 2]` (gid_global, arr) or per-cluster
  `assign_c [ci] [n_g]` int tensor of array indices (GPU).
- Route count `cnt()`:
  `used = zeros(n_arr, dp) bool`; for each cluster, for each (group, arr):
  `used[arr] |= outs_bool[group]`. Vectorise per cluster:
  `used.scatter_or_(arr_idx, outs_bool)` — implement scatter-OR via
  `used[arr_idx.unsqueeze(1).expand(-1,dp)] |= outs_bool` (gather-friendly).
  Return `used.sum()`.
- **Swap move** (Bug 4): pick cluster `c`, pick two arrays `e0, e1`.
  Find groups in cluster `c` currently at `e0` and `e1`; swap their array
  assignments. If one side empty, move a group. This matches Algorithm 1
  ("Swap(R_current, c, e_0, e_1)").
  - Vectorised neighbour eval: only the two arrays' rows in `used` change,
    so recompute just `used[e0]` and `used[e1]` after swap (delta update)
    instead of full `cnt()`. Track `curr_r` incrementally.
- **Acceptance** (Bug 5): compare against `best_r` per paper:
  `accept if nr < best_r or rand < exp((best_r - nr - 1)/T)`.
  - Keep `curr` and `best` separate; update `best` when `nr < best_r`.
- ITERS: paper says "> 10^5". Use `max(10000, 50 * n_arr * dp // 64)` so
  large layers get more iterations (paper: iterations proportional to
  initial connections). Cap at 200000.
- Return `(best_placement, best_r, history)`.

### Cell "Compile All Layers" — fix LUT INIT layout (Bug 8), weights sign (Bug 9)

**Weight sign** (Bug 9):
- `get_integer_weights()` returns unsigned `0..2^n-1` (offset binary).
- For LUT MAC we need the actual signed weight value. Subtract the offset:
  `w_signed = qi.to(int32) - (2**(n-1) - 1)`? No — N2UQ quantises to
  `2^n - 1` levels over `[-clip/2, clip/2]`, so levels are
  `-(2^(n-1)) .. +(2^(n-1))`? With `n=3`, `2^n-1 = 7` levels:
  `-3,-2,-1,0,1,2,3` (7 values, symmetric). The `qi = round((cw+clip/2)*n)`
  maps `cw ∈ [-clip/2, clip/2]` to `qi ∈ [0, 2^n-1] = [0,7]`, so signed =
  `qi - 3` (i.e. `qi - (2^n-1)/2` = `qi - 3`). Use
  `w_signed = qi.to(torch.int32) - ((2**n - 1) // 2)`.
  - Verify: N2UQ uses `gamma = (2^n-1)/(2^(n-1))` so for n=3, gamma=7/4=1.75,
    and the integer levels after rounding are 0..7, centre 3.5 — but levels
    are symmetric integers -3..3, so the offset is 3 (floor(7/2)). Confirm
    against `resnet.py` and `test_accuracy.ipynb` emulator.
- Carry `w_signed` through `extract_weight_groups` so `gid_map` keys are
  signed tuples, and LUT MAC uses signed arithmetic.

**LUT INIT layout** (Bug 8):
- After clustering + SA, for each array `ai in [0, n_arr)`:
  - For each select code `s in [0, N_clus)` (= cluster index):
    - Find the weight group that cluster `s` placed at array `ai`:
      invert `best_placement[s]` (which is `{gid: arr}`) to get
      `group_at[s][ai] = gid` (or `zero_gid` if none).
  - Build `wg_mat [N_clus, G]` of signed weights for those groups.
  - For all 64 addresses: `sel = addr >> G; act_bits = addr & 0x7`;
    `mac[s] = sum_g act_bit_g * wg_mat[sel, g]`.
  - `mac` is the partial sum over one bit-serial cycle (G activation bits,
    each 0/1). Range: `[-3*max|w|, +3*max|w|]`. Encode as 2's complement in
    `N_lut = 5` bits.
  - Pack `N_lut` INIT values (64-bit each), one per output bit:
    `init[k] = sum_{addr} ((mac[addr] >> k) & 1) << addr`, with negative
    `mac` in 2's complement (`mac & ((1<<N_lut)-1)`).
- Write **one file per array** `lut_arr{ai}.hex` with exactly `N_lut` lines
  (one 64-bit INIT per output bit). This matches `lut_array.sv INIT_FILE`.
- Remove the outer `for ci` loop — clusters are encoded via the select bits,
  not duplicated per array.

**Vectorise INIT generation**:
- For a layer, build `group_at [n_arr, N_clus]` (gid per array×select).
- Gather signed weights: `wg_all [n_arr, N_clus, G]` from `id2g`.
- `sel_idx = (addr_range >> G) & (N_clus-1)  -> [64]`
- `abits = addr_range & 0x7 -> [64]`; `act_bits [64, G]`.
- `sel_wg = wg_all[:, sel_idx, :]  -> [n_arr, 64, G]`
- `mac = (act_bits * sel_wg).sum(-1)  -> [n_arr, 64]`
- `mac_u = mac & ((1<<N_lut)-1)` (2's complement, handles negatives)
- Pack: for k in range(N_lut): `init[n_arr, k] = (mac_u >> k) & 1` then
  fold over addresses with bit-shift. Use:
  `bits = (mac_u.unsqueeze(-1) >> torch.arange(N_lut)) & 1  # [n_arr, 64, N_lut]`
  `init = (bits * (1 << torch.arange(64))).sum(1)  # [n_arr, N_lut]`
  Fully vectorised, no Python loops over addresses.

### Cell "Export TLMAC Hex Files" — fix Bug 7, 10

**`cluster_map.hex`** (Bug 10):
- One entry per step `s in [0, D_s)`, value = cluster index (0..N_clus-1),
  which equals `wg_sel`. N_clus=8 -> 3 bits -> **1 hex digit**.
- Write `f"{stc[s]:X}\n"` (single digit).

**`lut_arr{ai}.hex`**:
- `N_lut` lines per file, each 64-bit INIT. `f"{init[ai,k]:016X}\n"`.

**`routing_bitmap.hex`** (Bug 7):
- Match `switch_network.sv ROUTING_BITMAP [OUT_CHANNELS*ARR-1:0]`.
- Decide ordering: RTL param is flat `[D_p * N_arr - 1 : 0]`. Use
  `bit[arr * D_p + p]` (array-major) OR `bit[p * N_arr + arr]` (output-major).
  - Check `switch_network.sv` TODO — it's a stub, so we define the convention.
  - Document choice in metadata.json. Use output-major
    (`bit = p * N_arr + arr`) so slicing per output channel is contiguous
    (matches "each output linked to a subset of arrays" in §4).
- `R [n_arr, D_p] bool` -> flatten as `R.T.reshape(-1)` (output-major).
- Write as `(D_p * N_arr + 63)//64` 64-bit words, one per line.

**`metadata.json`**: add `D_p`, `D_s` (corrected), `weight_offset`,
  `routing_bitmap_order: "output_major"`, `wg_sel_bits`.

## Open questions
1. **N2UQ weight sign / offset** (investigated, needs Colab verify):
   - `resnet.py` `HardQuantizeConv`: `qi = round((cw + clip/2) * n)` where
     `n = (2^N-1)/clip`, `cw ∈ [-clip/2, clip/2]` (scaled weights).
   - For N=3, clip=2: `n = 3.5`, so `(cw+1)*3.5 ∈ [0, 7]`, rounded to
     `{0,1,2,3,4,5,6,7}` = **8 values** (0..2^N-1).
   - N2UQ paper claims `2^N - 1 = 7` uniform *levels*, but the offset-binary
     encoding produces 8 codes. The symmetric integer weight is
     `w_signed = qi - (2^N-1)/2 = qi - 3.5`? Non-integer — so the actual
     stored levels are `{0..7}` mapped to `{-3.5,-2.5,...,+3.5}` only if we
     subtract 3.5, but N2UQ rounds to integers so the dequantised weights
     are `sf * (round(...)/n - clip/2)` = `sf * (qi/3.5 - 1)`.
   - **For TLMAC**: the LUT stores the *integer* MAC `sum_g a_g^b * w_g` where
     `w_g` is the **signed integer** weight. The correct signed integer is
     `w_int = qi - (2^(N-1))`? With N=3: `qi - 4 ∈ {-4..+3}`. OR
     `w_int = qi - 3 ∈ {-3..+4}`. **Need to verify on Colab** by checking
     `torch.unique(qi)` on a real loaded layer and matching against the
     dequantised `sf*(qi/n - clip/2)` to pick the offset that makes the LUT
     MAC equal the software conv. **Action**: in the notebook, after loading,
     print `torch.unique(module.get_integer_weights())` and assert the offset.
   - **Decision for now**: use `w_signed = qi.to(int32) - ((2**N - 1) // 2)` =
     `qi - 3` (gives `{-3..+4}`, 8 levels). Document that this is pending
     Colab verification against the N2UQ dequant. The LUT MAC then uses these
     signed integers directly; 2's complement in `N_lut` bits handles negatives.
   - Also fix `get_integer_weights()` in **both** notebooks to optionally
     return signed (`qi - offset`) so the emulator and compiler agree.
2. **D_o not divisible by 64**: ResNet-18 has D_o ∈ {64,128,256,512}, all
   divisible by 64. Add an assert + a note; no padding needed for this model.
3. **Spectral clustering on GPU for D_s=2048**: `torch.linalg.eigh` on a
   2048x2048 float32 symmetric matrix is ~fast on GPU. If Colab GPU OOMs,
   fall back to `torch.linalg.eigh` on CPU (still vectorised) or
   `sklearn.cluster.SpectralClustering`. Document the fallback.
4. **Routing bitmap bit order**: `switch_network.sv` is a TODO stub, so we
   define the convention. Confirm with the RTL implementer (or just document).
5. **SA iterations scaling**: paper says "> 10^5" and "proportional to
   initial connections". Use `max(10000, 50*n_arr*dp//64)` capped at 200k.

## Migration / verification steps
1. On Colab, load a layer and print `torch.unique(module.get_integer_weights())`
   to confirm the weight offset (OQ1). Set `WEIGHT_OFFSET = (2**N-1)//2`.
2. ✅ Rewrote helpers cell: `extract_weight_groups` (correct D_s/D_p),
   `spectral_cluster_torch` (eigh + Cluster QR), `routing_sa` (per-cluster
   route count, array-swap move, best_r acceptance). All on `device`.
3. ✅ Rewrote compile cell: signed weights, vectorised INIT generation
   (`group_at [n_arr, NCLUS]`, one file per array, N_lut lines,
   address = {sel, act_bits}).
4. ✅ Rewrote export cell: 1-digit cluster_map, output-major routing
   bitmap, metadata fields.
5. ⏳ Rewrite `test_accuracy.ipynb` `TLMACConvEmulator` to match new layout
   (D_s/D_p, one file per array, signed result). NOT YET DONE.
6. ⏳ Run on Colab (GPU). Verify (NOT YET DONE):
   - `D_s, D_p` match paper formula for each layer (D_p=192, D_s=D_i*D_o/64).
   - `N_arr` per layer is in the same ballpark as Figure 5 bars (3-bit).
   - LUT-6 total = `sum(n_arr * N_lut)` per layer (NOT `* N_clus`).
   - TLMACConvEmulator reloads hex and matches software quantised conv.

## Non-goals
- Implementing the actual RTL (TODO stubs).
- Higher-level dataflow / tiling / HLS layers.
- Changing N2UQ architecture or training.
