#include <nano_ep/pipeline.hpp>

#include "dispatch_main.cuh"

namespace nano_ep {

// IPC ownership and exact receive allocation stay in the core extension.
static const DispatchMainSpecialization specialization{NANO_EP_HIDDEN,
                                                       NANO_EP_TOPK,
                                                       NANO_EP_WORLD_SIZE,
                                                       NANO_EP_EXPERTS_PER_RANK,
                                                       NANO_EP_MAX_TOKENS,
                                                       NANO_EP_EXPERT_ALIGNMENT,
                                                       &prepare_dispatch_launch<NANO_EP_HIDDEN,
                                                                                NANO_EP_TOPK,
                                                                                NANO_EP_WORLD_SIZE,
                                                                                NANO_EP_EXPERTS_PER_RANK,
                                                                                NANO_EP_MAX_TOKENS,
                                                                                NANO_EP_EXPERT_ALIGNMENT>,
                                                       &launch_dispatch<NANO_EP_HIDDEN,
                                                                        NANO_EP_TOPK,
                                                                        NANO_EP_WORLD_SIZE,
                                                                        NANO_EP_EXPERTS_PER_RANK,
                                                                        NANO_EP_MAX_TOKENS,
                                                                        NANO_EP_EXPERT_ALIGNMENT>};

}  // namespace nano_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("launcher") = pybind11::capsule(&nano_ep::specialization, "nano_ep.DispatchMainSpecialization");
    module.attr("geometry") = pybind11::make_tuple(
        NANO_EP_HIDDEN, NANO_EP_TOPK, NANO_EP_WORLD_SIZE, NANO_EP_EXPERTS_PER_RANK, NANO_EP_MAX_TOKENS, NANO_EP_EXPERT_ALIGNMENT);
}
