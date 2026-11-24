#pragma once

// Cluaiz fork only: C API that upstream llama.cpp does not have. See CLUAIZ_PATCHES.md.

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    // Advisory OS prefetch of the host (mmap) experts that the MoE cache of ctx is about to upload or
    // compute on the CPU, issued right before the upload, at most max_bytes per upload. 0 = off (default).
    // Has no effect on a context without a MoE cache. Set 0 before llama_free(ctx) so the entry is dropped.
    LLAMA_API void llama_cluaiz_set_moe_prefetch(struct llama_context * ctx, size_t max_bytes);

    // For model loads on the calling thread after this call: true leaves the weights that stay in CPU memory
    // mapped from the file (the CPU's default buffer type, e.g. MoE experts kept on the CPU) out of the
    // load-time OS prefetch (Linux MAP_POPULATE / POSIX_MADV_WILLNEED, Windows PrefetchVirtualMemory); they
    // are read on first use. They are passed to llama_mmap as lazy ranges, so on Linux they also get
    // POSIX_MADV_RANDOM. false (default) = upstream behaviour. Meant for a model larger than free RAM.
    LLAMA_API void llama_cluaiz_set_lazy_host_weights(bool on);

#ifdef __cplusplus
}
#endif
