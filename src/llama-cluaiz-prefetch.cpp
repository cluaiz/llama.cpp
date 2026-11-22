// Cluaiz fork only (CLUAIZ_PATCHES.md): advisory prefetch of the host experts the MoE cache is about to read.

#include "llama-cluaiz.h"
#include "llama-cluaiz-prefetch.h"

#include "llama-context.h"
#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace {

struct prefetch_state {
    size_t cap    = 0;
    size_t calls  = 0;
    size_t ranges = 0;
    size_t bytes  = 0;
};

// keyed by the GPU backends of a context: the MoE cache copy() only knows its backend
std::mutex g_mutex;
std::unordered_map<ggml_backend_t, prefetch_state> g_state;
std::atomic<size_t> g_n_on{0}; // lets copy() skip the lock while no context has a cap

// Linux: one MADV_WILLNEED reads at most one readahead window, whatever the range length (llama.cpp PR #32515), so hint each window
#if defined(__linux__)
constexpr size_t readahead_window = 128*1024;
#else
constexpr size_t readahead_window = 0; // Windows: all ranges in one PrefetchVirtualMemory call
#endif

bool is_gpu(ggml_backend_t backend) {
    const auto type = ggml_backend_dev_type(ggml_backend_get_device(backend));
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

}

size_t llama_cluaiz_moe_prefetch_cap(ggml_backend_t backend) {
    if (g_n_on.load(std::memory_order_relaxed) == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_state.find(backend);
    return it == g_state.end() ? 0 : it->second.cap;
}

void llama_cluaiz_moe_prefetch_ranges(ggml_backend_t backend, llama_memory_ranges && mr, size_t bytes) {
    if (mr.empty()) {
        return;
    }
    const size_t n_ranges = mr.size();
    if (readahead_window == 0) {
        llama_prefetch(std::move(mr));
    } else {
        for (const auto & range : mr) {
            for (size_t off = 0; off < range.size; off += readahead_window) {
                llama_prefetch({ { (const uint8_t *) range.addr + off, std::min(readahead_window, range.size - off) } });
            }
        }
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_state.find(backend);
    if (it == g_state.end()) {
        return;
    }
    prefetch_state & st = it->second;
    if (st.calls == 0) {
        LLAMA_LOG_INFO("llama_cluaiz_moe_prefetch: first prefetch: %zu ranges, %zu KiB\n", n_ranges, bytes/1024);
    }
    st.calls  += 1;
    st.ranges += n_ranges;
    st.bytes  += bytes;
}

void llama_cluaiz_set_moe_prefetch(llama_context * ctx, size_t max_bytes) {
    if (ctx == nullptr) {
        return;
    }
    ggml_backend_sched_t sched = ctx->get_sched();
    if (sched == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    prefetch_state total;
    bool was_on = false;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        if (!is_gpu(backend)) {
            continue;
        }
        if (max_bytes > 0) {
            g_state[backend].cap = max_bytes;
            continue;
        }
        const auto it = g_state.find(backend);
        if (it != g_state.end()) {
            was_on = true;
            total.calls  += it->second.calls;
            total.ranges += it->second.ranges;
            total.bytes  += it->second.bytes;
            g_state.erase(it);
        }
    }
    g_n_on.store(g_state.size(), std::memory_order_relaxed);
    // whole numbers only: the Cluaiz log drops lines that contain a dot
    if (max_bytes > 0) {
        LLAMA_LOG_INFO("llama_cluaiz_moe_prefetch: on, at most %zu MiB per MoE cache upload\n", max_bytes/1024/1024);
    } else if (was_on) {
        LLAMA_LOG_INFO("llama_cluaiz_moe_prefetch: off: %zu calls, %zu ranges, %zu MiB prefetched\n", total.calls, total.ranges, total.bytes/1024/1024);
    }
}
