#pragma once

#include <torch/extension.h>

#include <vector>

namespace nano_ep {

// All inputs/outputs use the current CUDA device and current PyTorch stream.
std::vector<torch::Tensor> route_counts(const torch::Tensor& topk_idx, int64_t num_experts, int64_t world_size);
torch::Tensor pack_rows(const torch::Tensor& x, const torch::Tensor& indices);
torch::Tensor expand_rows(const torch::Tensor& x, const torch::Tensor& expanded_slots, int64_t capacity);
torch::Tensor reduce_rows(const torch::Tensor& x, const torch::Tensor& expanded_slots);

}  // namespace nano_ep
