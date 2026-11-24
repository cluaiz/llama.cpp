llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-memory.cpp
    llama-moe-cache.cpp
    llama-cluaiz-prefetch.cpp # cluaiz: fork-only, see CLUAIZ_PATCHES.md
    llama-cluaiz-load.cpp # cluaiz: fork-only, see CLUAIZ_PATCHES.md
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

# cluaiz: stop the build when a merge dropped the MoE expert prefetch call (CLUAIZ_PATCHES.md)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/llama-moe-cache.cpp CLUAIZ_MOE_PREFETCH_CALL REGEX "llama_cluaiz_moe_prefetch[(]backend")
if (NOT CLUAIZ_MOE_PREFETCH_CALL)
    message(FATAL_ERROR "cluaiz: src/llama-moe-cache.cpp lost the llama_cluaiz_moe_prefetch call in copy(); re-apply it as CLUAIZ_PATCHES.md says")
endif()

# cluaiz: stop the build when a merge dropped the load prefetch ranges call (CLUAIZ_PATCHES.md)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/llama-model-loader.cpp CLUAIZ_LOAD_RANGES_CALL REGEX "llama_cluaiz_mmap_lazy_ranges[(][*]this, idx[)]")
if (NOT CLUAIZ_LOAD_RANGES_CALL)
    message(FATAL_ERROR "cluaiz: src/llama-model-loader.cpp lost the llama_cluaiz_mmap_lazy_ranges call in init_mappings(); re-apply it as CLUAIZ_PATCHES.md says")
endif()

add_library(llama
            ../include/llama.h
            ${LLAMA_CORE_SOURCES}
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ../include/llama.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

target_include_directories(llama PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

find_package(Threads REQUIRED)
target_link_libraries(llama PUBLIC ggml PRIVATE Threads::Threads)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
