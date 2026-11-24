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

Windows asks each expert slice once per context (since run 60, 2026-11-24): `PrefetchVirtualMemory` runs on the
calling thread and costs about 43 us per MiB for pages already in the working set and about 85 us per MiB for
pages on the standby list, and saves nothing for them; only pages read from the disk gain (run 60
micro-benchmark, `test/cloud/runs/2026-11-24-run60-windows-prefetch-cost/`). Run 44 asked 173-215 GB per run for 11.5 GB of experts.
`llama-cluaiz-prefetch.cpp` keeps the addresses already asked in the context's entry and skips them (`ask_once`,
`_WIN32` only); the cap now counts only the ranges really asked. Linux is unchanged (asks every time). The end
line reports the skipped MiB.

Known limit: `llama_prefetch` logs one warning per failed call (upstream behaviour, not changed here). A slice
evicted after it was asked is read by page faults again on Windows (as with the prefetch off).

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

## 2. Earlier fork-only commits

From `git log e574a17b0..17d1465f0 --no-merges` (2026-11-22). `e574a17b0` is the upstream side of the last
merge (`90a3d489b`, upstream at "model : metal : workaround fusion for LFM3 (#32296)", 2026-11-15). What each
commit still changes today was read with `git show <commit>` and `git diff e574a17b0 HEAD -- <file>`
(2026-11-23). These commits have no `cluaiz:` markers; the `git grep` above does not find them.

| Commit | Title | Still different from upstream today | Re-apply |
|---|---|---|---|
| `55551986f` | PTX BFE CUDA dequantization and RMS-Norm numerical clamp | no (reverted by `4dfb0b68c`) | nothing |
| `447ab54f6` | kernel optimizations (BitNet AVX2, M-RoPE KV cache, cached CUDA context, legacy GGUF mapping) | only the M-RoPE part in `src/llama-kv-cache.cpp` (the rest reverted by `4f51802a5` and `4dfb0b68c`) | section 2.2 |
| `d81e1b3ff` | expose C-ABI for native thinking tags extraction and prompt templating | yes, `common/chat.{h,cpp}` (its `src/llama-sampler.cpp` change was reverted by `4dfb0b68c`) | section 2.1 |
| `4f51802a5` | restore default GGML_OP_OFFLOAD_MIN_BATCH to 32 | no: `ggml/src/ggml-cuda/ggml-cuda.cu` equals upstream | nothing |
| `4dfb0b68c` | revert unverified kernel hacks and numerical overrides | no: its 5 files equal upstream | nothing |
| `44cf982da` | persistent templates handle to eliminate per-message re-parsing | yes, `common/chat.{h,cpp}` | section 2.1 |
| `8f9842cd1` | model: support embeddinggemma2 (text+vision+audio) | no: a backport of upstream PR #30054, which upstream has as `4fbc76dec` (2026-10-06); its 10 files equal upstream | nothing |
| `17d1465f0` | render the C ABI prompt with common_chat_prompt::str() | yes, `common/chat.cpp` | section 2.1 |

`git diff --stat e574a17b0 HEAD` lists only these fork files: `CLUAIZ_PATCHES.md`, `common/chat.cpp` (+147),
`common/chat.h` (+55), `src/llama-kv-cache.cpp` (+6 -2), and the section 1 files.

### 2.1 Chat C ABI (`d81e1b3ff`, `44cf982da`, `17d1465f0`)

What it does: plain C functions over upstream's chat template code in `common/`, so the Engine (Rust FFI,
`engine/llama/src/ffi/llama_cpp.rs`) can render a prompt with the model's own Jinja template and read its
thinking tags without C++ types.

| File | Change |
|---|---|
| `common/chat.h` | appended at the end of the file, in one `extern "C" { }` block: the `LLAMA_COMMON_API` macro (`__declspec(dllexport)` on Windows, `visibility("default")` elsewhere) and the declarations of the five functions below, plus the opaque `struct llama_chat_templates_handle` |
| `common/chat.cpp` | appended at the end of the file, in one `extern "C" { }` block: the five functions and the static helper `llama_chat_render_with_templates` |

The functions:

- `llama_chat_extract_thinking_tags(model, custom_tmpl, out_start, max_start_len, out_end, max_end_len)`:
  `common_chat_templates_init`, then `common_chat_templates_apply` with `enable_thinking = true` and one user
  message "hi"; when the result has `supports_thinking`, copies `thinking_start_tag` and the first of
  `thinking_end_tags`. Returns false on any failure (exceptions are caught).
- `llama_chat_apply_template_native(model, custom_tmpl, roles, contents, n_messages, add_ass, enable_thinking, buf, buf_size)`:
  parses the template, then renders through `llama_chat_render_with_templates`.
- `llama_chat_templates_handle_init(model, custom_tmpl)` / `llama_chat_templates_handle_free(handle)`: the
  handle holds a `common_chat_templates_ptr`, parsed once per loaded model (`44cf982da`). It must be freed
  before the model. NULL on failure.
- `llama_chat_apply_template_handle(handle, roles, contents, n_messages, add_ass, enable_thinking, buf, buf_size)`:
  same output as `llama_chat_apply_template_native`, without parsing again.
- `llama_chat_render_with_templates` (static): fills `common_chat_templates_inputs` (`enable_thinking`,
  `add_generation_prompt = add_ass`, one `common_chat_msg` per role/content), calls
  `common_chat_templates_apply`, takes `params.prompt.str()` (`17d1465f0`: upstream's chat refactor made
  `common_chat_params::prompt` a list of parts), copies at most `buf_size - 1` bytes plus a NUL and returns the
  full prompt length, or -1 on an exception.

Engine callers: `engine/llama/src/native/templater.rs` (handle init, render with the handle or the native
call, thinking tags), `engine/llama/src/native/core.rs` (handle free), `engine/runtime/src/host.rs` (thinking
tags).

Re-apply after an upstream merge:

1. Both blocks are appended at the end of the files, so a merge conflicts only when upstream also adds at the
   end. Keep upstream's new lines and put the `extern "C"` block after them again.
2. Check that the upstream names the code uses still exist in `common/chat.h`: `common_chat_templates_init`,
   `common_chat_templates_apply`, `common_chat_templates_ptr`, `common_chat_templates_inputs` with
   `enable_thinking`, `add_generation_prompt` and `messages`, `common_chat_msg` with `role` and `content`,
   `common_chat_params` with `prompt` (and `common_chat_prompt::str()`), `supports_thinking`,
   `thinking_start_tag`, `thinking_end_tags`. If one changed, adapt the C function body; keep the C signatures,
   which the Rust declarations in `engine/llama/src/ffi/llama_cpp.rs` mirror.
3. Build the Engine (`cargo build -p cmd`). A missing symbol fails the link. Check the exports:
   `grep -c llama_chat_apply_template_handle target/debug/llama-common.dll` gives 1 (2026-11-23).
4. Live check: one chat answer through the Engine with a thinking model (the prompt is rendered and the
   thinking tags are found).

### 2.2 M-RoPE KV shift (`447ab54f6`, the part that is left)

What it does: lets the KV cache move positions (context shift, cache reuse) for models with M-RoPE
(`n_pos_per_embd() > 1`), which upstream refuses.

`src/llama-kv-cache.cpp`, three places:

1. `llama_kv_cache::seq_add`: the line
   `GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");`
   is commented out, with the note `// Bypass for M-RoPE multimodal models`.
2. `llama_kv_cache::seq_div`: the same for its `seq_div()` assert.
3. `llama_kv_cache::get_can_shift`: inside `if (hparams.n_pos_per_embd() > 1) {`, before `return false;`,
   `if (hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE) { return true; }`
   with the comment `// Allow shifting for M-RoPE models (e.g. Qwen2-VL, Qwen3.5-VL)`.

Effect in the Engine: `llama_memory_can_shift` returns true for these models, so `engine/llama/src/native/stream.rs`
may shift or reuse the cache. The Engine still turns both off for prompts with media, for quantized keys and
when `llama_memory_can_shift` is false (`stream.rs`, `can_shift`).

Not checked in this note: whether the shifted positions are numerically right for M-RoPE models (upstream's
asserts say the operation is not supported). No test of it was found or run for this note.

Re-apply after an upstream merge:

1. If the three places conflict, take upstream's version and redo the three edits above. Find them with
   `git grep -n "n_pos_per_embd" -- src/llama-kv-cache.cpp`.
2. Check that `LLAMA_ROPE_TYPE_MROPE` and `LLAMA_ROPE_TYPE_IMROPE` still exist in `include/llama.h`.
3. Check: `git diff <upstream commit> HEAD -- src/llama-kv-cache.cpp` shows exactly these three hunks
   (+6 -2), then build the Engine (`cargo build -p cmd`).

## 3. Load prefetch skip for CPU-mapped weights (off by default)

What it does: at load, `llama_mmap` asks the OS to read the whole file except upstream's lazy ranges (Linux
`MAP_POPULATE` + `POSIX_MADV_WILLNEED`, Windows `PrefetchVirtualMemory`, `src/llama-mmap.cpp`). When the
application calls `llama_cluaiz_set_lazy_host_weights(true)` on the loading thread, the fork adds to those lazy
ranges every weight that stays mapped from the file in the CPU's default buffer type (the condition
`is_default_buft` in `src/llama-model.cpp`, e.g. MoE experts kept on the CPU), unless the same weight is also in
another context (copied at load). Those pages are then read on first use. Upstream's own mechanism does the rest:
with lazy ranges, no `MAP_POPULATE`; `WILLNEED` / `PrefetchVirtualMemory` on the complement only; Linux also
`POSIX_MADV_RANDOM` on the lazy ranges. Nothing else of the lazy-tensor feature (`lazy_read`, row gathering) is
used. Off unless the setter is called, so with the setter unused this fork behaves like upstream.

Why not upstream's settings: upstream has no setting for the load prefetch (`init_mappings(true, ...)` is fixed
in `src/llama-model.cpp`), and its `lazy_mode` marks only tensors the architecture flags as `TENSOR_READ_LAZY`
(per-layer token embeddings) and moves them to a row-gather path, which experts must not take (checked in
`competitor-repos/llama.cpp` at `d3c3e24`, 2026-11-16).

| File | Kind | Change |
|---|---|---|
| `include/llama-cluaiz.h` | fork only | `llama_cluaiz_set_lazy_host_weights(bool)`, thread-local, default false |
| `src/llama-cluaiz-load.h`, `src/llama-cluaiz-load.cpp` | new, fork only | the setter and `llama_cluaiz_mmap_lazy_ranges(ml, idx)`: `ml.lazy.for_file(idx)` plus the CPU-mapped weight ranges when on; one log line `llama_cluaiz_lazy_host_weights: file N: M MiB of CPU-mapped weights left out of the load prefetch` |
| `src/llama-model-loader.cpp` | upstream, 2 lines | `#include "llama-cluaiz-load.h"` after `#include "llama-model-loader.h"`; in `init_mappings()` the last argument of `std::make_unique<llama_mmap>(...)` is `llama_cluaiz_mmap_lazy_ranges(*this, idx)` instead of `lazy.for_file(idx)` |
| `src/CMakeLists.txt` | upstream, 2 places | `llama-cluaiz-load.cpp` in `LLAMA_CORE_SOURCES` after `llama-cluaiz-prefetch.cpp`; a second guard block before `add_library(llama` that stops the configure step when the call is missing from `llama-model-loader.cpp` |

### Re-apply after an upstream merge

1. If `src/llama-model-loader.cpp` conflicts, keep upstream's version and re-add the two marked lines. If
   upstream renamed `lazy.for_file`, change `llama-cluaiz-load.cpp` to call the new name.
2. Check that the names the fork file uses still exist: `llama_model_loader::{lazy, use_mmap, ctx_map,
   get_weight}`, `ctx_key::buft`, `llama_tensor_weight::{idx, offs}`, `llama_mmap::ranges`, and that the CPU
   default buffer type is still the one condition for wrapping the mapping (`is_default_buft` in
   `llama-model.cpp`). If the mapping condition changed, follow it in `llama-cluaiz-load.cpp`.
3. Build (`cargo build -p cmd`); the guard fails the configure step if the call is gone.
4. Live check: a load with `CLUAIZ_LAZY_HOST_WEIGHTS=on` and a model larger than free RAM prints the log line
   above with a non-zero MiB (`engine/llama/README.md`).
