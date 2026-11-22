#pragma once

// Cluaiz fork only (CLUAIZ_PATCHES.md): advisory prefetch of the host experts the MoE cache is about to read.
// Off unless llama_cluaiz_set_moe_prefetch (include/llama-cluaiz.h) set a cap for the context.

#include "llama-mmap.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// the cap set for the context that owns backend, 0 = off
size_t llama_cluaiz_moe_prefetch_cap(ggml_backend_t backend);

// prefetches mr (bytes in total, already within the cap) with llama_prefetch and counts it for backend
void llama_cluaiz_moe_prefetch_ranges(ggml_backend_t backend, llama_memory_ranges && mr, size_t bytes);

// called by llama_moe_cache copy() before its upload loop: the selected experts of every fill, then of every
// CPU miss, of each host expert tensor, until the cap is reached; F has an int32_t member `expert`
template <typename F>
void llama_cluaiz_moe_prefetch(ggml_backend_t backend, const std::vector<ggml_tensor *> & experts,
        const std::vector<F> & fills, const std::vector<F> & cpu) {
    if (fills.empty() && cpu.empty()) {
        return;
    }
    const size_t cap = llama_cluaiz_moe_prefetch_cap(backend);
    if (cap == 0) {
        return;
    }
    llama_memory_ranges mr;
    size_t bytes = 0;
    auto add = [&](const std::vector<F> & list) {
        for (const F & f : list) {
            for (const ggml_tensor * w : experts) {
                if (w->data == nullptr || w->buffer == nullptr || !ggml_backend_buffer_is_host(w->buffer)) {
                    continue;
                }
                const size_t size = w->nb[2];
                if (bytes + size > cap) {
                    return false;
                }
                mr.push_back({ (const uint8_t *) w->data + (size_t) f.expert*size, size });
                bytes += size;
            }
        }
        return true;
    };
    if (add(fills)) {
        add(cpu);
    }
    llama_cluaiz_moe_prefetch_ranges(backend, std::move(mr), bytes);
}
