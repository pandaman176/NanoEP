#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/Event.h>
#include <torch/extension.h>

#include <memory>
#include <vector>

namespace nano_ep {

struct CompletionEvent {
    std::shared_ptr<c10::Event> event;
    std::vector<torch::Tensor> tensors;

    CompletionEvent(const c10::cuda::CUDAStream& stream, std::vector<torch::Tensor> outputs)
        : event(std::make_shared<c10::Event>(c10::DeviceType::CUDA)), tensors(std::move(outputs)) {
        event->record(stream);
    }

    void wait(const c10::cuda::CUDAStream& stream) const {
        stream.unwrap().wait(*event);
        for (const auto& tensor : tensors)
            tensor.record_stream(stream);
    }

    void current_stream_wait() const { wait(c10::cuda::getCurrentCUDAStream()); }
};

}  // namespace nano_ep
