#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdio>
#include <nano_ep/ops.hpp>

namespace nano_ep {

template <int kNumRanks>
__global__ void route_counts_kernel(
    const int64_t* topk_idx, int64_t* rank_counts, int64_t* expert_counts, int64_t num_tokens, int64_t num_topk, int64_t num_experts) {
    const int64_t token_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token_id >= num_tokens)
        return;

    const int64_t experts_per_rank = num_experts / kNumRanks;
    bool seen[kNumRanks] = {};

    for (int64_t k = 0; k < num_topk; ++k) {
        const int64_t expert_id = topk_idx[token_id * num_topk + k];
        if (expert_id < -1 || expert_id >= num_experts) {
            printf("NanoEP route_counts invalid expert: token=%lld expert=%lld\n",
                   static_cast<long long>(token_id),
                   static_cast<long long>(expert_id));
            asm volatile("trap;");
            return;
        }
        if (expert_id == -1)
            continue;
        const int64_t rank_id = expert_id / experts_per_rank;
        atomicAdd(reinterpret_cast<unsigned long long*>(&expert_counts[expert_id]), 1ULL);
        seen[rank_id] = true;
    }

    for (int64_t rank_id = 0; rank_id < kNumRanks; ++rank_id) {
        if (seen[rank_id]) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&rank_counts[rank_id]), 1ULL);
        }
    }
}

template <typename data_t>
__global__ void packed_rows_kernel(
    const data_t* x, const int64_t* indices, data_t* packed_rows, int64_t num_tokens, int64_t num_rows, int64_t hidden_dim) {
    const int64_t element_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t row_id = element_id / hidden_dim;
    if (row_id >= num_rows)
        return;

    const int64_t token_id = indices[row_id];
    if (token_id < 0 || token_id >= num_tokens)
        return;

    packed_rows[element_id] = x[token_id * hidden_dim + (element_id % hidden_dim)];
}

template <typename data_t>
__global__ void expanded_rows_kernel(const data_t* x,
                                     const int64_t* expanded_slots,
                                     data_t* expanded_rows,
                                     int64_t hidden_dim,
                                     int64_t num_slots_per_row,
                                     int64_t num_rows,
                                     int64_t capacity) {
    const int64_t row_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row_id >= num_rows)
        return;
    for (int64_t k = 0; k < num_slots_per_row; ++k) {
        const int64_t slot_id = expanded_slots[row_id * num_slots_per_row + k];
        if (slot_id < 0 || slot_id >= capacity)
            continue;

        for (int64_t d = 0; d < hidden_dim; ++d) {
            expanded_rows[slot_id * hidden_dim + d] = x[row_id * hidden_dim + d];
        }
    }
}

template <typename data_t>
__global__ void reduced_rows_kernel(const data_t* x,
                                    const int64_t* expanded_slots,
                                    data_t* reduced_rows,
                                    int64_t hidden_dim,
                                    int64_t num_slots_per_row,
                                    int64_t num_rows,
                                    int64_t capacity) {
    using acc_t = at::acc_type<data_t, /*is_cuda=*/true>;

    const int64_t element_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t row_id = element_id / hidden_dim;
    const int64_t hidden_id = element_id % hidden_dim;
    if (row_id >= num_rows)
        return;

    acc_t acc{};
    for (int64_t k = 0; k < num_slots_per_row; ++k) {
        const int64_t slot_id = expanded_slots[row_id * num_slots_per_row + k];
        if (slot_id < 0 || slot_id >= capacity)
            continue;
        acc += static_cast<acc_t>(x[slot_id * hidden_dim + hidden_id]);
    }
    reduced_rows[element_id] = static_cast<data_t>(acc);
}

#define NANO_LAUNCH_ROUTE_COUNTS(R)                                                                    \
    route_counts_kernel<R><<<blocks, threads, 0, stream.stream()>>>(topk_idx.data_ptr<int64_t>(),      \
                                                                    rank_counts.data_ptr<int64_t>(),   \
                                                                    expert_counts.data_ptr<int64_t>(), \
                                                                    num_tokens,                        \
                                                                    num_topk,                          \
                                                                    num_experts)

std::vector<torch::Tensor> route_counts(const torch::Tensor& topk_idx, int64_t num_experts, int64_t world_size) {
    // Input Check:
    // topk_idx: [T, K] int64 tensor of top-k expert indices for each token
    // num_experts: number of experts
    // world_size: number of ranks
    TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64, "topk_idx must be of type int64");
    TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must be a 2D tensor");
    TORCH_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous");
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");
    TORCH_CHECK(world_size == 1 || world_size == 2 || world_size == 4 || world_size == 8 || world_size == 16 || world_size == 32,
                "world_size must be one of {1, 2, 4, 8, 16, 32}, got ",
                world_size);
    TORCH_CHECK(num_experts % world_size == 0, "num_experts must be divisible by world_size");

    // CUDA Guard
    c10::cuda::CUDAGuard device_guard(topk_idx.device());

    // Initialize Output:
    // rank_counts: [world_size] int64 tensor of counts of tokens routed to each rank
    // expert_counts: [num_experts] int64 tensor of counts of tokens routed to each expert
    auto rank_counts = torch::zeros({world_size}, topk_idx.options());
    auto expert_counts = torch::zeros({num_experts}, topk_idx.options());

    const int64_t num_tokens = topk_idx.size(0);
    const int64_t num_topk = topk_idx.size(1);

    if (num_tokens == 0) {
        return {rank_counts, expert_counts};
    }

    auto stream = c10::cuda::getCurrentCUDAStream();

    constexpr int threads = 256;
    const int64_t blocks = (num_tokens + threads - 1) / threads;
    switch (world_size) {
        case 1:
            NANO_LAUNCH_ROUTE_COUNTS(1);
            break;
        case 2:
            NANO_LAUNCH_ROUTE_COUNTS(2);
            break;
        case 4:
            NANO_LAUNCH_ROUTE_COUNTS(4);
            break;
        case 8:
            NANO_LAUNCH_ROUTE_COUNTS(8);
            break;
        case 16:
            NANO_LAUNCH_ROUTE_COUNTS(16);
            break;
        case 32:
            NANO_LAUNCH_ROUTE_COUNTS(32);
            break;
        default:
            TORCH_CHECK(false, "Unsupported world_size: ", world_size);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {rank_counts, expert_counts};
}
#undef NANO_LAUNCH_ROUTE_COUNTS

torch::Tensor pack_rows(const torch::Tensor& x, const torch::Tensor& indices) {
    // input check
    //  x: [N, D] bf16/float32/float64 tensor of input rows
    //  indices: [M] int64 tensor of row indices to pack
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat64,
                "x must be of type bfloat16, float32, or float64");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(indices.dim() == 1, "indices must be a 1D tensor");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be of type int64");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(indices.device() == x.device(), "indices and x must be on the same CUDA device");

    // CUDA Guard
    c10::cuda::CUDAGuard device_guard(x.device());

    // Output:
    // packed_rows: [M, D] tensor of packed rows
    auto packed_rows = torch::empty({indices.size(0), x.size(1)}, x.options());

    const int64_t num_tokens = x.size(0);
    const int64_t hidden_dim = x.size(1);
    const int64_t num_rows = indices.size(0);

    if (packed_rows.numel() == 0) {
        return packed_rows;
    }

    auto stream = c10::cuda::getCurrentCUDAStream();

    const int64_t elements = num_rows * hidden_dim;
    constexpr int threads = 256;
    const int64_t blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, x.scalar_type(), "nano_ep::pack_rows", [&] {
        packed_rows_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            x.data_ptr<scalar_t>(), indices.data_ptr<int64_t>(), packed_rows.data_ptr<scalar_t>(), num_tokens, num_rows, hidden_dim);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return packed_rows;
}
torch::Tensor expand_rows(const torch::Tensor& x, const torch::Tensor& expanded_slots, int64_t capacity) {
    // Input Check:
    // x: [N, H] tensor of input rows
    // expanded_slots: [N, K] int64 tensor of expanded slot indices for each row
    // capacity: maximum number of expanded rows
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat64,
                "x must be of type bfloat16, float32, or float64");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(expanded_slots.is_cuda(), "expanded_slots must be a CUDA tensor");
    TORCH_CHECK(expanded_slots.dim() == 2, "expanded_slots must be a 2D tensor");
    TORCH_CHECK(expanded_slots.scalar_type() == torch::kInt64, "expanded_slots must be of type int64");
    TORCH_CHECK(expanded_slots.is_contiguous(), "expanded_slots must be contiguous");
    TORCH_CHECK(expanded_slots.size(0) == x.size(0), "expanded_slots must have the same number of rows as x");
    TORCH_CHECK(expanded_slots.device() == x.device(), "expanded_slots and x must be on the same CUDA device");
    TORCH_CHECK(capacity >= 0, "capacity must be a non-negative integer");

    // CUDA Guard
    c10::cuda::CUDAGuard device_guard(x.device());

    // Output:
    // expanded_rows: [capacity, H] tensor of expanded rows
    auto expanded_rows = torch::zeros({capacity, x.size(1)}, x.options());

    const int64_t hidden_dim = x.size(1);
    const int64_t num_slots_per_row = expanded_slots.size(1);
    const int64_t num_rows = x.size(0);

    if (expanded_rows.numel() == 0 || num_rows == 0 || num_slots_per_row == 0) {
        return expanded_rows;
    }

    auto stream = c10::cuda::getCurrentCUDAStream();

    constexpr int threads = 256;
    const int64_t blocks = (num_rows + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, x.scalar_type(), "nano_ep::expand_rows", [&] {
        expanded_rows_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(x.data_ptr<scalar_t>(),
                                                                                expanded_slots.data_ptr<int64_t>(),
                                                                                expanded_rows.data_ptr<scalar_t>(),
                                                                                hidden_dim,
                                                                                num_slots_per_row,
                                                                                num_rows,
                                                                                capacity);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return expanded_rows;
}
torch::Tensor reduce_rows(const torch::Tensor& x, const torch::Tensor& expanded_slots) {
    // Input Check:
    // x: [capacity, H] tensor of expert outputs
    // expanded_slots: [N, K] int64 tensor of expanded slot indices for each row
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat64,
                "x must be of type bfloat16, float32, or float64");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(expanded_slots.is_cuda(), "expanded_slots must be a CUDA tensor");
    TORCH_CHECK(expanded_slots.dim() == 2, "expanded_slots must be a 2D tensor");
    TORCH_CHECK(expanded_slots.scalar_type() == torch::kInt64, "expanded_slots must be of type int64");
    TORCH_CHECK(expanded_slots.is_contiguous(), "expanded_slots must be contiguous");
    TORCH_CHECK(expanded_slots.device() == x.device(), "expanded_slots and x must be on the same CUDA device");

    // CUDA Guard
    c10::cuda::CUDAGuard device_guard(x.device());

    // Output:
    // reduced_rows: [N, H] tensor of reduced rows
    auto reduced_rows = torch::zeros({expanded_slots.size(0), x.size(1)}, x.options());

    const int64_t hidden_dim = x.size(1);
    const int64_t num_slots_per_row = expanded_slots.size(1);
    const int64_t num_rows = expanded_slots.size(0);
    const int64_t capacity = x.size(0);

    if (reduced_rows.numel() == 0 || num_rows == 0 || num_slots_per_row == 0) {
        return reduced_rows;
    }

    auto stream = c10::cuda::getCurrentCUDAStream();

    const int64_t elements = num_rows * hidden_dim;
    constexpr int threads = 256;
    const int64_t blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, x.scalar_type(), "nano_ep::reduce_rows", [&] {
        reduced_rows_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(x.data_ptr<scalar_t>(),
                                                                               expanded_slots.data_ptr<int64_t>(),
                                                                               reduced_rows.data_ptr<scalar_t>(),
                                                                               hidden_dim,
                                                                               num_slots_per_row,
                                                                               num_rows,
                                                                               capacity);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return reduced_rows;
}

}  // namespace nano_ep
