#include <nano_ep/pipeline.hpp>

#include "combine_main.cuh"

namespace nano_ep {

// Core extension owns validation, IPC lifetime and barriers. This module only
// launches the return kernel for its compiled geometry, on the supplied stream.
static const CombineMainSpecialization specialization{
    NANO_EP_HIDDEN,
    NANO_EP_TOPK,
    NANO_EP_WORLD_SIZE,
    NANO_EP_EXPERTS_PER_RANK,
    NANO_EP_MAX_TOKENS,
    &launch_combine_return<NANO_EP_HIDDEN, NANO_EP_TOPK, NANO_EP_WORLD_SIZE, NANO_EP_MAX_TOKENS>};

}  // namespace nano_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("launcher") = pybind11::capsule(&nano_ep::specialization, "nano_ep.CombineMainSpecialization");
    module.attr("geometry") =
        pybind11::make_tuple(NANO_EP_HIDDEN, NANO_EP_TOPK, NANO_EP_WORLD_SIZE, NANO_EP_EXPERTS_PER_RANK, NANO_EP_MAX_TOKENS);
}
