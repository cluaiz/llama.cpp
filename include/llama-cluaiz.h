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

#ifdef __cplusplus
}
#endif
