#include <cstring>
#include <nano_ep/ops.hpp>
#include <nano_ep/peer.hpp>
#include <nano_ep/pipeline.hpp>
#include <string>

#include "layout.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<nano_ep::DispatchState>(module, "DispatchState")
        .def(py::init<>())
        .def_readwrite("metadata", &nano_ep::DispatchState::metadata)
        .def_readwrite("counts", &nano_ep::DispatchState::counts)
        .def_readwrite("stage_buffer", &nano_ep::DispatchState::stage_buffer)
        .def_readwrite("hidden", &nano_ep::DispatchState::hidden)
        .def_readwrite("x_dtype", &nano_ep::DispatchState::x_dtype)
        .def_readonly("rank", &nano_ep::DispatchState::rank)
        .def_readonly("metadata_prepared", &nano_ep::DispatchState::metadata_prepared);
    module.def("dispatch_main",
               &nano_ep::dispatch_main,
               py::arg("runtime"),
               py::arg("x"),
               py::arg("indices"),
               py::arg("weights"),
               py::arg("num_experts"),
               py::arg("max_tokens"),
               py::arg("alignment"),
               py::arg("cached_metadata"),
               py::arg("allow_multiple_reduction") = false);
    module.def("notify_counts", &nano_ep::notify_counts);
    module.def("build_receive_metadata",
               &nano_ep::build_receive_metadata,
               py::arg("stage_buffer"),
               py::arg("expert_offsets"),
               py::arg("rank"),
               py::arg("experts_per_rank"),
               py::arg("num_topk"));
    module.def(
        "reserve_and_put",
        [](nano_ep::PeerRuntime& runtime,
           const torch::Tensor& x,
           const torch::Tensor& indices,
           const c10::optional<torch::Tensor>& weights,
           int64_t num_experts,
           int64_t max_tokens,
           const c10::optional<torch::Tensor>& cached_dst_slots,
           bool allow_multiple_reduction) {
            TORCH_CHECK(x.dim() == 2 && indices.dim() == 2, "x and indices must be 2D");
            const auto layout = nano_ep::WindowLayout::make(
                runtime.world_size(), max_tokens, x.size(1), indices.size(1), num_experts, allow_multiple_reduction);
            return nano_ep::reserve_and_put(runtime, x, indices, weights, num_experts, max_tokens, layout, cached_dst_slots);
        },
        py::arg("runtime"),
        py::arg("x"),
        py::arg("indices"),
        py::arg("weights"),
        py::arg("num_experts"),
        py::arg("max_tokens"),
        py::arg("cached_dst_slots") = py::none(),
        py::arg("allow_multiple_reduction") = false);
    module.def("dispatch_epilogue", &nano_ep::dispatch_epilogue);
    module.def("combine_main",
               &nano_ep::combine_main,
               py::arg("runtime"),
               py::arg("x"),
               py::arg("metadata"),
               py::arg("weights"),
               py::arg("max_tokens"),
               py::arg("expand"),
               py::arg("allow_multiple_reduction"));
    module.def("combine_epilogue",
               &nano_ep::combine_epilogue,
               py::arg("stage_buffer"),
               py::arg("source_topk_snapshot"),
               py::arg("hidden"),
               py::arg("x_dtype"),
               py::arg("num_experts"),
               py::arg("world_size"),
               py::arg("max_tokens"),
               py::arg("bias"),
               py::arg("with_weights"),
               py::arg("expand"),
               py::arg("allow_multiple_reduction"));
    module.def(
        "window_layout",
        [](int ranks, int64_t tokens, int hidden, int topk, int64_t num_experts, bool allow_multiple_reduction) {
            const auto x = nano_ep::WindowLayout::make(ranks, tokens, hidden, topk, num_experts, allow_multiple_reduction);
            return std::vector<int64_t>{x.signal_offset, x.count_offset, x.payload_offset, x.return_offset, x.num_bytes};
        },
        py::arg("ranks"),
        py::arg("tokens"),
        py::arg("hidden"),
        py::arg("topk"),
        py::arg("num_experts"),
        py::arg("allow_multiple_reduction") = false);
    module.def("route_counts", &nano_ep::route_counts);
    module.def("pack_rows", &nano_ep::pack_rows);
    module.def("expand_rows", &nano_ep::expand_rows);
    module.def("reduce_rows", &nano_ep::reduce_rows);
    py::class_<nano_ep::PeerRuntime>(module, "PeerRuntime")
        .def(py::init<int, int, int64_t>())
        .def("export_local_handle",
             [](const nano_ep::PeerRuntime& runtime) {
                 const auto bytes = runtime.export_local_handle();
                 return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
             })
        .def("import_peer_handles",
             [](nano_ep::PeerRuntime& runtime, const std::vector<py::bytes>& handles) {
                 std::vector<nano_ep::IpcHandleBytes> native_handles(handles.size());
                 for (size_t i = 0; i < handles.size(); ++i) {
                     auto& bytes = native_handles[i];
                     if (py::len(handles[i]) != bytes.size()) {
                         throw py::value_error("Each IPC handle must contain exactly " + std::to_string(bytes.size()) + " bytes");
                     }
                     std::memcpy(bytes.data(), PyBytes_AS_STRING(handles[i].ptr()), bytes.size());
                 }
                 runtime.import_peer_handles(native_handles);
             })
        .def("peer_addresses", &nano_ep::PeerRuntime::peer_addresses)
        .def("local_window", &nano_ep::PeerRuntime::local_window)
        .def("barrier", py::overload_cast<int64_t>(&nano_ep::PeerRuntime::barrier))
        .def("barrier", py::overload_cast<int64_t, int64_t>(&nano_ep::PeerRuntime::barrier))
        .def("close_peer_mappings", &nano_ep::PeerRuntime::close_peer_mappings)
        .def("close", &nano_ep::PeerRuntime::close);
}
