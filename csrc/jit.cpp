#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <compare>
#include <cstdlib>
#include <deep_jit/backend/cuda/backend.hpp>
#include <format>
#include <map>
#include <mutex>
#include <nano_ep/jit.hpp>
#include <utility>

#include "layout.hpp"

namespace nano_ep {

using Runtime = deep_jit::Runtime<deep_jit::CUDA>;

enum class KernelKind { DispatchMain, DispatchEpilogue, CombineMain, CombineEpilogue };

struct KernelSpec {
    KernelKind kind;
    int64_t hidden, topk;
    int64_t world_size = 0, experts_per_rank = 0, max_tokens = 0, alignment = 1;
    bool cached = false, expand = false, multiple = false, weights = false, zero_padding = false;

    auto operator<=>(const KernelSpec&) const = default;
};

struct JitKernel {
    Runtime& runtime;
    std::shared_ptr<deep_jit::cuda::Kernel> kernel;
    KernelKind kind;
    int threads, smem_bytes;
    bool cooperative;
};

namespace {

namespace driver = deep_jit::cuda::driver;

// Each CUDA device has its own context-bound kernel handles.
struct DeviceKernels {
    std::unique_ptr<Runtime> runtime;
    std::map<KernelSpec, JitKernel> kernels;
};

std::mutex jit_mutex;
std::map<int, DeviceKernels> devices;
std::filesystem::path source_dir;

struct KernelLaunchPlan {
    int threads, smem_bytes;
    bool cooperative;
};

struct KernelSource {
    const char* name;
    std::string arguments;
    KernelLaunchPlan plan;
};

const JitKernel& prepare_kernel(const KernelSpec& spec);

// Host and device code use the same shared-memory sizing formulas.
KernelSource describe(const KernelSpec& spec) {
    const auto [kind, hidden, topk, ranks, experts, tokens, alignment, cached, expand, multiple, weights, zero_padding] = spec;
    const auto* device = at::cuda::getCurrentDeviceProperties();
    switch (kind) {
        case KernelKind::DispatchMain: {
            const auto plan = make_dispatch_payload_plan(hidden, topk, ranks, experts, dispatch_notify_warps(cached));
            TORCH_CHECK(plan.payload_warps > 0, "dispatch main cannot fit one staged row on this device");
            return {"dispatch_main",
                    std::format("{}, {}, {}, {}, {}, {}, {}", hidden, topk, ranks, experts, tokens, alignment, cached),
                    {plan.total_warps * 32, static_cast<int>(plan.num_bytes), true}};
        }
        case KernelKind::DispatchEpilogue: {
            const auto plan = PayloadRowPlan::make(hidden, topk);
            const int warps = std::min<int64_t>(plan.total_warps, device->sharedMemPerBlockOptin / plan.stage_bytes);
            return {"dispatch_epilogue",
                    std::format("{}, {}, {}, {}, {}, {}, {}", hidden, topk, experts, cached, expand, weights, zero_padding),
                    {warps * 32, static_cast<int>(warps * plan.stage_bytes), false}};
        }
        case KernelKind::CombineMain: {
            const auto plan = HiddenRowPlan::make(hidden, true);
            const int warps = plan.max_warps();
            return {"combine_main",
                    std::format("{}, {}, {}, {}, nv_bfloat16, {}, {}, {}", hidden, topk, ranks, tokens, expand, multiple, weights),
                    {warps * 32, static_cast<int>(warps * plan.stage_bytes), true}};
        }
        case KernelKind::CombineEpilogue: {
            const auto plan = HiddenRowPlan::make(hidden);
            const int warps = plan.max_warps();
            return {"combine_epilogue",
                    std::format("{}, {}, {}, {}, {}, {}, {}, {}", hidden, topk, ranks, experts, tokens, expand, multiple, weights),
                    {warps * 32, static_cast<int>(warps * plan.stage_bytes), false}};
        }
    }
    TORCH_CHECK(false, "Unknown NanoEP kernel kind");
}

}  // namespace

void init_jit(const std::string& path) {
    source_dir = std::filesystem::absolute(path);
}

const JitKernel& dispatch_main_kernel(
    int64_t hidden, int64_t topk, int64_t ranks, int64_t experts, int64_t tokens, int64_t alignment, bool cached) {
    return prepare_kernel({KernelKind::DispatchMain, hidden, topk, ranks, experts, tokens, cached ? 1 : alignment, cached});
}

const JitKernel& dispatch_epilogue_kernel(
    int64_t hidden, int64_t topk, int64_t experts, bool cached, bool expand, bool weights, bool zero_padding) {
    return prepare_kernel(
        {KernelKind::DispatchEpilogue, hidden, topk, 0, experts, 0, 1, cached, expand, false, weights, expand && zero_padding});
}

const JitKernel& combine_main_kernel(
    int64_t hidden, int64_t topk, int64_t ranks, int64_t experts, int64_t tokens, bool expand, bool multiple, bool weights) {
    return prepare_kernel({KernelKind::CombineMain, hidden, topk, ranks, 0, tokens, 1, false, expand, multiple, weights});
}

const JitKernel& combine_epilogue_kernel(
    int64_t hidden, int64_t topk, int64_t ranks, int64_t experts, int64_t tokens, bool expand, bool multiple, bool weights) {
    return prepare_kernel({KernelKind::CombineEpilogue, hidden, topk, ranks, experts, tokens, 1, false, expand, multiple, weights});
}

void prepare_dispatch(int64_t hidden,
                      int64_t topk,
                      int64_t ranks,
                      int64_t experts,
                      int64_t tokens,
                      int64_t alignment,
                      bool cached,
                      bool expand,
                      bool weights,
                      bool zero_padding) {
    dispatch_main_kernel(hidden, topk, ranks, experts, tokens, alignment, cached);
    dispatch_epilogue_kernel(hidden, topk, experts, cached, expand, weights, zero_padding);
}

void prepare_combine(
    int64_t hidden, int64_t topk, int64_t ranks, int64_t experts, int64_t tokens, bool expand, bool multiple, bool weights) {
    combine_main_kernel(hidden, topk, ranks, experts, tokens, expand, multiple, weights);
    combine_epilogue_kernel(hidden, topk, ranks, experts, tokens, expand, multiple, weights);
}

namespace {

const JitKernel& prepare_kernel(const KernelSpec& spec) {
    // DeepJIT releases the GIL during compilation. A waiting Python caller must
    // release it before waiting for the thread that holds the cache mutex.
    std::unique_lock lock(jit_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        deep_jit::GilScopedRelease gil_release;
        lock.lock();
    }
    const int device = c10::cuda::current_device();
    auto& cache = devices[device];
    if (auto entry = cache.kernels.find(spec); entry != cache.kernels.end())
        return entry->second;
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream().stream(), &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "Warm this NanoEP kernel before CUDA Graph capture");
    if (!cache.runtime) {
        cache.runtime = std::make_unique<Runtime>(deep_jit::Config(
            source_dir,
            "NANO_EP",
            {},
            {source_dir, source_dir / "include"},
            {"nano_ep/", "layout.hpp", "dispatch_main.cuh", "dispatch_epilogue.cuh", "combine_main.cuh", "combine_epilogue.cuh"}));
        cache.runtime->default_compiler_options.optimize_level = "2";
        cache.runtime->default_compiler_options.with_line_info = true;
    }
    const auto [name, arguments, plan] = describe(spec);
    TORCH_CHECK(plan.threads > 0, name, " cannot fit one staged row on this device");
    const auto source = std::format(R"(
#include <{0}.cuh>

static void instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&nano_ep::{0}_kernel<{1}>);
}}
)",
                                    name,
                                    arguments);
    auto kernel = cache.runtime->compile(name, source);
    if (plan.cooperative) {
        int supported = 0;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device));
        TORCH_CHECK(supported, name, " requires CUDA cooperative launch support");
        DJ_CUDA_DRIVER_CHECK(
            driver::lazy_cuFuncSetAttribute(kernel->kernel_handle, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, plan.smem_bytes));
        int active_blocks = 0;
        // The cooperative grids use at most one block per SM.
        static const auto occupancy = reinterpret_cast<decltype(&cuOccupancyMaxActiveBlocksPerMultiprocessor)>(
            dlsym(driver::get_cuda_handle(), "cuOccupancyMaxActiveBlocksPerMultiprocessor"));
        TORCH_CHECK(occupancy, "CUDA driver does not expose cooperative occupancy queries");
        DJ_CUDA_DRIVER_CHECK(occupancy(&active_blocks, kernel->kernel_handle, plan.threads, plan.smem_bytes));
        TORCH_CHECK(active_blocks > 0, name, " cannot fit one staged block per SM");
    }
    return cache.kernels
        .emplace(spec, JitKernel{*cache.runtime, std::move(kernel), spec.kind, plan.threads, plan.smem_bytes, plan.cooperative})
        .first->second;
}

}  // namespace

bool use_pdl() {
    const auto* disable_pdl = std::getenv("NANO_EP_DISABLE_PDL");
    return at::cuda::getCurrentDeviceProperties()->major >= 9 && !(disable_pdl && std::string_view(disable_pdl) == "1");
}

template <size_t... I>
void launch(const JitKernel& kernel, const deep_jit::cuda::LaunchOptions& options, std::span<void*> args, std::index_sequence<I...>) {
    TORCH_INTERNAL_ASSERT(args.size() == sizeof...(I));
    kernel.runtime.launch(kernel.kernel, options, deep_jit::NoRefPtr{args[I]}...);
}

void launch_kernel(const JitKernel& kernel, cudaStream_t stream, int sms, bool pdl, std::span<void*> args) {
    const deep_jit::cuda::LaunchOptions options{.stream = stream,
                                                .num_smem_bytes = kernel.smem_bytes,
                                                .grid_dim = dim3(sms),
                                                .block_dim = dim3(kernel.threads),
                                                .cooperative = kernel.cooperative,
                                                .enable_pdl = pdl};
    switch (kernel.kind) {
        case KernelKind::DispatchMain:
            return launch(kernel, options, args, std::make_index_sequence<16>{});
        case KernelKind::DispatchEpilogue:
            return launch(kernel, options, args, std::make_index_sequence<14>{});
        case KernelKind::CombineMain:
            return launch(kernel, options, args, std::make_index_sequence<9>{});
        case KernelKind::CombineEpilogue:
            return launch(kernel, options, args, std::make_index_sequence<7>{});
    }
}

}  // namespace nano_ep
