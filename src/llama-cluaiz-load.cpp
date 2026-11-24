// Cluaiz fork only (CLUAIZ_PATCHES.md): leave the weights that stay in CPU memory mapped from the file out of
// the load-time OS prefetch, so a model larger than free RAM does not read its whole file at load.

#include "llama-cluaiz.h"
#include "llama-cluaiz-load.h"

#include "llama-impl.h"
#include "llama-model-loader.h"

#include "ggml-backend.h"

#include <unordered_set>

namespace {

// per loading thread: llama_model_load_from_file runs init_mappings on the thread that called it
thread_local bool g_lazy_host_weights = false;

}

void llama_cluaiz_set_lazy_host_weights(bool on) {
    g_lazy_host_weights = on;
}

llama_mmap::ranges llama_cluaiz_mmap_lazy_ranges(const llama_model_loader & ml, uint32_t idx) {
    llama_mmap::ranges ranges = ml.lazy.for_file(idx);
    if (!g_lazy_host_weights || !ml.use_mmap) {
        return ranges;
    }
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        return ranges;
    }
    // only the CPU's default buffer type wraps the mapping itself (llama-model.cpp, is_default_buft); a
    // tensor in any other context (GPU, pinned host, repack) is copied at load and keeps the prefetch
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_dev_buffer_type(cpu_dev);
    std::unordered_set<const llama_model_loader::llama_tensor_weight *> copied;
    for (const auto & it : ml.ctx_map) {
        if (!it.second || it.first.buft == cpu_buft) {
            continue;
        }
        for (ggml_tensor * t = ggml_get_first_tensor(it.second.get()); t; t = ggml_get_next_tensor(it.second.get(), t)) {
            copied.insert(ml.get_weight(ggml_get_name(t)));
        }
    }
    size_t bytes = 0;
    for (const auto & it : ml.ctx_map) {
        if (!it.second || it.first.buft != cpu_buft) {
            continue;
        }
        for (ggml_tensor * t = ggml_get_first_tensor(it.second.get()); t; t = ggml_get_next_tensor(it.second.get(), t)) {
            const llama_model_loader::llama_tensor_weight * w = ml.get_weight(ggml_get_name(t));
            if (w == nullptr || w->idx != idx || copied.count(w)) {
                continue;
            }
            ranges.emplace_back(w->offs, w->offs + ggml_nbytes(t));
            bytes += ggml_nbytes(t);
        }
    }
    // whole numbers only: the Cluaiz log drops lines that contain a dot
    LLAMA_LOG_INFO("llama_cluaiz_lazy_host_weights: file %u: %zu MiB of CPU-mapped weights left out of the load prefetch\n",
            idx, bytes/1024/1024);
    return ranges;
}
