#include <cstring>
#include <nano_ep/jit.hpp>
#include <nano_ep/peer.hpp>
#include <nano_ep/pipeline.hpp>
#include <string>

#include "layout.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("init_jit", &nano_ep::init_jit);
    module.def("prepare_dispatch", &nano_ep::prepare_dispatch);
    module.def("prepare_combine", &nano_ep::prepare_combine);
    module.def("dispatch",
               &nano_ep::dispatch,
               py::arg("runtime"),
               py::arg("x"),
               py::arg("indices"),
               py::arg("weights"),
               py::arg("num_experts"),
               py::arg("max_tokens"),
               py::arg("alignment"),
               py::arg("cached_metadata"),
               py::arg("cached_expanded_capacity"),
               py::arg("allow_multiple_reduction"),
               py::arg("expand"),
               py::arg("zero_padding"),
               py::arg("num_sms"),
               py::arg("comm_stream"),
               py::arg("previous_event") = -1,
               py::arg("async_op") = true);
    module.def("combine",
               &nano_ep::combine,
               py::arg("runtime"),
               py::arg("x"),
               py::arg("metadata"),
               py::arg("weights"),
               py::arg("bias"),
               py::arg("max_tokens"),
               py::arg("expand"),
               py::arg("allow_multiple_reduction"),
               py::arg("num_sms"),
               py::arg("comm_stream"),
               py::arg("previous_event") = -1,
               py::arg("async_op") = true);
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
    py::class_<nano_ep::CompletionEvent, std::shared_ptr<nano_ep::CompletionEvent>>(module, "CompletionEvent")
        .def("current_stream_wait", &nano_ep::CompletionEvent::current_stream_wait);
}
