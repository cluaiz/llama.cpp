#pragma once

// Cluaiz fork only (CLUAIZ_PATCHES.md): which parts of a model file the load-time OS prefetch skips.
// Off unless llama_cluaiz_set_lazy_host_weights (include/llama-cluaiz.h) was called on the loading thread.

#include "llama-mmap.h"

#include <cstdint>

struct llama_model_loader;

// the ranges of file idx that llama_mmap leaves out of its load-time prefetch: upstream's lazy ranges, plus,
// when the calling thread turned it on, the weights that stay mapped from the file in the CPU's default
// buffer type (read on first use instead of at load)
llama_mmap::ranges llama_cluaiz_mmap_lazy_ranges(const llama_model_loader & ml, uint32_t idx);
