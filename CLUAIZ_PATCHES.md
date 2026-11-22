# Cluaiz fork patches

This fork (`cluaiz/llama.cpp`, branch `cluaiz-main`) is upstream `ggml-org/llama.cpp` plus the changes below. Each one
lists the upstream lines it touches and how to re-apply it after an upstream merge. Fork-only code lives in files
whose names contain `cluaiz`, which upstream never touches, so a merge conflicts only on the marked lines.

Find every marked upstream line with:

```
git grep -n "cluaiz:" -- src include ggml common CMakeLists.txt
```

## 1. MoE expert prefetch (advisory, off by default)

What it does: right before the MoE cache (`src/llama-moe-cache.cpp`, `copy()`) uploads the experts a layer
selected, it asks the OS to read the selected host (mmap) experts of every fill and every CPU miss in one call,
with upstream's own `llama_prefetch` (`src/llama-mmap.cpp`). The addresses are the cache's own
(`w->data + expert*nb[2]`). No thread, no buffer, no new cache policy, no backend code. A failed prefetch is only
a warning from `llama_prefetch`; the upload then reads the pages as before. Off unless the application calls the
setter, so with the setter unused this fork behaves like upstream.

| File | Kind | Change |
|---|---|---|
| `include/llama-cluaiz.h` | new, fork only | `llama_cluaiz_set_moe_prefetch(ctx, max_bytes)`, 0 = off |
| `src/llama-cluaiz-prefetch.h` | new, fork only | `llama_cluaiz_moe_prefetch` template called by `copy()`: collects the ranges up to the cap |
| `src/llama-cluaiz-prefetch.cpp` | new, fork only | setter, per-context cap table keyed by the context's GPU backends, Linux readahead-window steps, counters and log lines |
| `src/llama-moe-cache.cpp` | upstream, 2 lines | `#include "llama-cluaiz-prefetch.h"` after `#include "llama-moe-cache.h"`; the call `llama_cluaiz_moe_prefetch(backend, l.experts, fills, cpu);` in `impl::copy()` after the CPU map update and before the comment `// upload the missing experts ...` |
| `src/CMakeLists.txt` | upstream, 2 places | `llama-cluaiz-prefetch.cpp` in `LLAMA_CORE_SOURCES` after `llama-moe-cache.cpp`; the guard block before `add_library(llama` that stops the configure step when the call is missing from `llama-moe-cache.cpp` |

Why no new field in `llama_context_params`: upstream appends fields there, so every merge would conflict and the
hand-written Rust mirror of the struct would drift. The setter keeps the struct identical to upstream.

Why the table is keyed by backend: `copy()` only receives its backend, and `llama_context` keeps its MoE cache
private. The setter walks `ctx->get_sched()` backends (public scheduler API). The caller sets 0 before
`llama_free(ctx)` so a freed backend address never keeps a cap.

Why `llama_prefetch` itself is not changed: on Linux one `MADV_WILLNEED` reads at most one readahead window
whatever the range length (upstream PR #32515, open on 2026-11-22). The fork file calls `llama_prefetch` once per
128 KiB window on Linux instead of editing it. If #32515 (or a similar change) is merged upstream, drop the window
loop in `llama-cluaiz-prefetch.cpp` and pass the ranges in one call as on Windows.

Known limit: `llama_prefetch` logs one warning per failed call (upstream behaviour, not changed here).

### Re-apply after an upstream merge

1. Merge. If `src/llama-moe-cache.cpp` or `src/CMakeLists.txt` conflict, keep upstream's version and re-add the
   marked lines from the table above.
2. Check that `copy()` still has the names the call uses: `backend`, `l.experts` (host expert tensors, experts
   along dim 2), `fills` and `cpu` (vectors of a struct with `int32_t expert`). If a name or meaning changed, adapt
   the call or the template, never drop the call: the CMake guard fails the build when the call is gone.
3. Check that `llama_prefetch(llama_memory_ranges)` and `llama_context::get_sched()` still exist; the fork file
   does not compile otherwise.
4. Build the Engine (`cargo build -p cmd` in the Engine repo) and run its live check with the MoE cache on
   (`engine/llama/README.md`, section on the MoE expert prefetch).

## Earlier fork-only commits (not yet described in this file)

From `git log e574a17b0..17d1465f0 --no-merges` on 2026-11-22: `17d1465f0` fix(chat) C ABI prompt,
`8f9842cd1` embeddinggemma2, `44cf982da` persistent templates handle, `4dfb0b68c` revert of kernel hacks,
`4f51802a5` GGML_OP_OFFLOAD_MIN_BATCH back to 32, `d81e1b3ff` chat C ABI, `447ab54f6` kernel changes,
`55551986f` PTX BFE dequantization and RMS-Norm clamp. Their re-apply steps are still to be written.
