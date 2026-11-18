#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <span>
#include <string>

namespace nano_ep {

struct JitKernel;

void init_jit(const std::string& source_dir);
bool use_pdl();
void launch_kernel(const JitKernel& kernel, cudaStream_t stream, int sms, bool pdl, std::span<void*> args);

template <typename... Args>
void launch_kernel(const JitKernel& kernel, cudaStream_t stream, int sms, bool pdl, const Args&... args) {
    void* storage[] = {const_cast<void*>(static_cast<const void*>(&args))...};
    launch_kernel(kernel, stream, sms, pdl, std::span<void*>(storage));
}

const JitKernel& dispatch_main_kernel(
    int64_t hidden, int64_t topk, int64_t world_size, int64_t experts_per_rank, int64_t max_tokens, int64_t alignment, bool cached);
const JitKernel& dispatch_epilogue_kernel(
    int64_t hidden, int64_t topk, int64_t experts_per_rank, bool cached, bool expand, bool weights, bool zero_padding);

const JitKernel& combine_main_kernel(int64_t hidden,
                                     int64_t topk,
                                     int64_t world_size,
                                     int64_t experts_per_rank,
                                     int64_t max_tokens,
                                     bool expand,
                                     bool multiple,
                                     bool weights);
const JitKernel& combine_epilogue_kernel(int64_t hidden,
                                         int64_t topk,
                                         int64_t world_size,
                                         int64_t experts_per_rank,
                                         int64_t max_tokens,
                                         bool expand,
                                         bool multiple,
                                         bool weights);

void prepare_dispatch(int64_t hidden,
                      int64_t topk,
                      int64_t world_size,
                      int64_t experts_per_rank,
                      int64_t max_tokens,
                      int64_t alignment,
                      bool cached,
                      bool expand,
                      bool weights,
                      bool zero_padding);

void prepare_combine(int64_t hidden,
                     int64_t topk,
                     int64_t world_size,
                     int64_t experts_per_rank,
                     int64_t max_tokens,
                     bool expand,
                     bool multiple,
                     bool weights);

}  // namespace nano_ep
