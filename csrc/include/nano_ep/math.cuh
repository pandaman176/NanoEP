#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nano_ep {

// All 32 lanes participate; each lane receives the sum of lanes [0, lane].
__device__ __forceinline__ int64_t warp_inclusive_sum(int64_t value, int lane) {
    constexpr int kWarpSize = 32;
#pragma unroll
    for (int delta = 1; delta < kWarpSize; delta *= 2) {
        const int64_t previous = __shfl_up_sync(0xffffffffu, value, delta);
        if (lane >= delta)
            value += previous;
    }
    return value;
}

}  // namespace nano_ep
