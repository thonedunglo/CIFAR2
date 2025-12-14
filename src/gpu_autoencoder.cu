#include "gpu_autoencoder.h"

#include <cuda_runtime.h>

#include <stdexcept>


__constant__ float c_biases_storage[2048];


namespace {
#define CUDA_CHECK(expr)                                                      \
    do {                                                                     \
        cudaError_t err__ = (expr);                                          \
        if (err__ != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(err__));             \
        }                                                                    \
    } while (0)

inline std::size_t act_size(std::size_t n, std::size_t c, std::size_t h,
                            std::size_t w) {
    return n * c * h * w;
}
}  // namespace

GPUAutoencoder::GPUAutoencoder(std::size_t batch_size_max)
    : batch_size_max_(batch_size_max) {
    // Weight sizes
    w1_size_ = 256 * 3 * 3 * 3;
    w2_size_ = 128 * 256 * 3 * 3;
    w3_size_ = 128 * 128 * 3 * 3;
    w4_size_ = 256 * 128 * 3 * 3;
    w5_size_ = 3 * 256 * 3 * 3;

    b1_size_ = 256;
    b2_size_ = 128;
    b3_size_ = 128;
    b4_size_ = 256;
    b5_size_ = 3;

    input_size_ = act_size(batch_size_max_, 3, 32, 32);
    act1_size_ = act_size(batch_size_max_, 256, 32, 32);
    act2_size_ = act_size(batch_size_max_, 256, 16, 16);
    act3_size_ = act_size(batch_size_max_, 128, 16, 16);
    act4_size_ = act_size(batch_size_max_, 128, 8, 8);
    act5_size_ = act_size(batch_size_max_, 128, 8, 8);
    act6_size_ = act_size(batch_size_max_, 128, 16, 16);
    act7_size_ = act_size(batch_size_max_, 256, 16, 16);
    act8_size_ = act_size(batch_size_max_, 256, 32, 32);
    act9_size_ = act_size(batch_size_max_, 3, 32, 32);

    alloc_weights_and_grads();
    alloc_activations(batch_size_max_);
}

GPUAutoencoder::~GPUAutoencoder() { free_all(); }

void GPUAutoencoder::alloc_weights_and_grads() {
    CUDA_CHECK(cudaMalloc(&d_w1_, w1_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2_, w2_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w3_, w3_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w4_, w4_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w5_, w5_size_ * sizeof(float)));


    // Thay vì cudaMalloc, ta lấy địa chỉ của mảng hằng số c_biases_storage
    float* d_const_base = nullptr;
    CUDA_CHECK(cudaGetSymbolAddress((void**)&d_const_base, c_biases_storage));

    d_b1_ = d_const_base; 
    d_b2_ = d_b1_ + b1_size_;
    d_b3_ = d_b2_ + b2_size_;
    d_b4_ = d_b3_ + b3_size_;
    d_b5_ = d_b4_ + b4_size_;


    CUDA_CHECK(cudaMalloc(&d_gw1_, w1_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw2_, w2_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw3_, w3_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw4_, w4_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw5_, w5_size_ * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_gb1_, b1_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb2_, b2_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb3_, b3_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb4_, b4_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb5_, b5_size_ * sizeof(float)));
}

void GPUAutoencoder::alloc_activations(std::size_t /*batch_size_max*/) {
    CUDA_CHECK(cudaMalloc(&d_act1_, act1_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act2_, act2_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act3_, act3_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act4_, act4_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act5_, act5_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act6_, act6_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act7_, act7_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act8_, act8_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_act9_, act9_size_ * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_g_act1_, act1_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_act2_, act2_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_act3_, act3_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_act4_, act4_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_act5_, act5_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_act6_, act6_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_act7_, act7_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_act8_, act8_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_act9_, act9_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_input_, input_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_, input_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1_idx_, act2_size_ * sizeof(uint32_t)));  // pool1 output size
    CUDA_CHECK(cudaMalloc(&d_pool2_idx_, act4_size_ * sizeof(uint32_t)));  // pool2 output size
    CUDA_CHECK(cudaMalloc(&d_loss_val_, sizeof(float))); // Allocate memory for scalar loss
}

void GPUAutoencoder::free_all() {
    auto freep = [](float*& p) {
        if (p) cudaFree(p);
        p = nullptr;
    };
    freep(d_w1_);
    freep(d_w2_);
    freep(d_w3_);
    freep(d_w4_);
    freep(d_w5_);
    
    // Vì chúng trỏ vào Constant Memory (được giải phóng tự động khi hết chương trình)
    d_b1_ = d_b2_ = d_b3_ = d_b4_ = d_b5_ = nullptr;

    freep(d_gw1_);
    freep(d_gw1_);
    freep(d_gw2_);
    freep(d_gw3_);
    freep(d_gw4_);
    freep(d_gw5_);
    freep(d_gb1_);
    freep(d_gb2_);
    freep(d_gb3_);
    freep(d_gb4_);
    freep(d_gb5_);

    freep(d_act1_);
    freep(d_act2_);
    freep(d_act3_);
    freep(d_act4_);
    freep(d_act5_);
    freep(d_act6_);
    freep(d_act7_);
    freep(d_act8_);
    freep(d_act9_);
    freep(d_g_act1_);
    freep(d_g_act2_);
    freep(d_g_act3_);
    freep(d_g_act4_);
    freep(d_g_act5_);
    freep(d_g_act6_);
    freep(d_g_act7_);
    freep(d_g_act8_);
    freep(d_g_act9_);
    freep(d_g_input_);
    freep(d_input_);
    if (d_pool1_idx_) cudaFree(d_pool1_idx_), d_pool1_idx_ = nullptr;
    if (d_pool2_idx_) cudaFree(d_pool2_idx_), d_pool2_idx_ = nullptr;
    if (d_loss_val_) cudaFree(d_loss_val_), d_loss_val_ = nullptr;    // Free scalar loss memory
}

void GPUAutoencoder::load_weights(const AutoencoderCPU& cpu) {
    CUDA_CHECK(cudaMemcpy(d_w1_, cpu.conv1().weights().data(),
                          w1_size_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2_, cpu.conv2().weights().data(),
                          w2_size_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w3_, cpu.conv3().weights().data(),
                          w3_size_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w4_, cpu.conv4().weights().data(),
                          w4_size_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w5_, cpu.conv5().weights().data(),
                          w5_size_ * sizeof(float), cudaMemcpyHostToDevice));
    
    size_t off1 = 0;
    size_t off2 = off1 + b1_size_ * sizeof(float);
    size_t off3 = off2 + b2_size_ * sizeof(float);
    size_t off4 = off3 + b3_size_ * sizeof(float);
    size_t off5 = off4 + b4_size_ * sizeof(float);

    CUDA_CHECK(cudaMemcpyToSymbol(c_biases_storage, cpu.conv1().bias().data(),
                                  b1_size_ * sizeof(float), off1));
    CUDA_CHECK(cudaMemcpyToSymbol(c_biases_storage, cpu.conv2().bias().data(),
                                  b2_size_ * sizeof(float), off2));
    CUDA_CHECK(cudaMemcpyToSymbol(c_biases_storage, cpu.conv3().bias().data(),
                                  b3_size_ * sizeof(float), off3));
    CUDA_CHECK(cudaMemcpyToSymbol(c_biases_storage, cpu.conv4().bias().data(),
                                  b4_size_ * sizeof(float), off4));
    CUDA_CHECK(cudaMemcpyToSymbol(c_biases_storage, cpu.conv5().bias().data(),
                                  b5_size_ * sizeof(float), off5));
}

void GPUAutoencoder::save_weights(AutoencoderCPU& cpu) const {
    CUDA_CHECK(cudaMemcpy(cpu.conv1().weights().data(), d_w1_,
                          w1_size_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu.conv2().weights().data(), d_w2_,
                          w2_size_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu.conv3().weights().data(), d_w3_,
                          w3_size_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu.conv4().weights().data(), d_w4_,
                          w4_size_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu.conv5().weights().data(), d_w5_,
                          w5_size_ * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(cpu.conv1().bias().data(), d_b1_,
                          b1_size_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu.conv2().bias().data(), d_b2_,
                          b2_size_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu.conv3().bias().data(), d_b3_,
                          b3_size_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu.conv4().bias().data(), d_b4_,
                          b4_size_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu.conv5().bias().data(), d_b5_,
                          b5_size_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPUAutoencoder::zero_grads() {
    CUDA_CHECK(cudaMemset(d_gw1_, 0, w1_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gw2_, 0, w2_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gw3_, 0, w3_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gw4_, 0, w4_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gw5_, 0, w5_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gb1_, 0, b1_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gb2_, 0, b2_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gb3_, 0, b3_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gb4_, 0, b4_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gb5_, 0, b5_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_act1_, 0, act1_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_act2_, 0, act2_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_act3_, 0, act3_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_act4_, 0, act4_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_act5_, 0, act5_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_act6_, 0, act6_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_act7_, 0, act7_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_act8_, 0, act8_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_act9_, 0, act9_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_input_, 0, input_size_ * sizeof(float)));
}

void GPUAutoencoder::copy_input_to_device(const std::vector<float>& host,
                                          std::size_t batch) {
    if (batch > batch_size_max_) {
        throw std::runtime_error("Batch size exceeds batch_size_max in copy_input_to_device");
    }
    const std::size_t elems = batch * 3 * 32 * 32;
    if (host.size() < elems) {
        throw std::runtime_error("Host input size too small for batch");
    }
    CUDA_CHECK(cudaMemcpy(d_input_, host.data(), elems * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// ===================== SGD update =====================
__global__ void sgd_update_kernel(float* param, const float* grad,
                                  std::size_t n, float lr) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        param[idx] -= lr * grad[idx];
    }
}

void GPUAutoencoder::step(float lr) {
    auto launch_update = [&](float* p, const float* g, std::size_t n) {
        const int block = 256;
        int grid = static_cast<int>((n + block - 1) / block);
        sgd_update_kernel<<<grid, block>>>(p, g, n, lr);
        CUDA_CHECK(cudaGetLastError());
    };
    launch_update(d_w1_, d_gw1_, w1_size_);
    launch_update(d_w2_, d_gw2_, w2_size_);
    launch_update(d_w3_, d_gw3_, w3_size_);
    launch_update(d_w4_, d_gw4_, w4_size_);
    launch_update(d_w5_, d_gw5_, w5_size_);
    launch_update(d_b1_, d_gb1_, b1_size_);
    launch_update(d_b2_, d_gb2_, b2_size_);
    launch_update(d_b3_, d_gb3_, b3_size_);
    launch_update(d_b4_, d_gb4_, b4_size_);
    launch_update(d_b5_, d_gb5_, b5_size_);
}

template <bool FuseReLU>
__global__ void conv2d_forward_kernel(const float* input, const float* weight,
                                      const float* bias, float* output,
                                      std::size_t batch, std::size_t in_c,
                                      std::size_t out_c, std::size_t height,
                                      std::size_t width) {
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t oc = blockIdx.z % out_c;
    const std::size_t n = blockIdx.z / out_c;
    if (n >= batch || y >= height || x >= width) return;

    auto idx4 = [&](std::size_t n, std::size_t c, std::size_t h, std::size_t w,
                    std::size_t C, std::size_t H, std::size_t W) {
        return ((n * C + c) * H + h) * W + w;
    };
    auto widx = [&](std::size_t oc, std::size_t ic, std::size_t kh,
                    std::size_t kw, std::size_t in_c) {
        return ((oc * in_c + ic) * 3 + kh) * 3 + kw;
    };

    float acc = bias[oc];
    for (std::size_t ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int in_y = static_cast<int>(y) + kh - 1;
                int in_x = static_cast<int>(x) + kw - 1;
                if (in_y < 0 || in_x < 0 || in_y >= static_cast<int>(height) ||
                    in_x >= static_cast<int>(width)) {
                    continue;
                }
                float inval = input[idx4(n, ic, static_cast<std::size_t>(in_y),
                                         static_cast<std::size_t>(in_x), in_c,
                                         height, width)];
                acc += inval * weight[widx(oc, ic, kh, kw, in_c)];
            }
        }
    }
    if (FuseReLU && acc < 0.0f) acc = 0.0f;
    output[idx4(n, oc, y, x, out_c, height, width)] = acc;
}

void conv2d_forward_naive(const float* input, const float* weight,
                          const float* bias, float* output, std::size_t batch,
                          std::size_t in_c, std::size_t out_c,
                          std::size_t height, std::size_t width) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
              batch * out_c);
    conv2d_forward_kernel<false><<<grid, block>>>(
        input, weight, bias, output, batch, in_c, out_c, height, width);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== Fused Conv + Bias + ReLU forward =====================
void conv2d_forward_relu(const float* input, const float* weight,
                         const float* bias, float* output,
                         std::size_t batch, std::size_t in_c,
                         std::size_t out_c, std::size_t height,
                         std::size_t width) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
              batch * out_c);
    conv2d_forward_kernel<true><<<grid, block>>>(
        input, weight, bias, output, batch, in_c, out_c, height, width);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== ReLU forward =====================
__global__ void relu_forward_kernel(const float* input, float* output,
                                    std::size_t total) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = input[idx];
        output[idx] = v > 0.0f ? v : 0.0f;
    }
}

void relu_forward_naive(const float* input, float* output, std::size_t total) {
    const int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    relu_forward_kernel<<<grid, block>>>(input, output, total);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== MaxPool 2x2 forward =====================
__global__ void maxpool2x2_forward_kernel(const float* input, float* output,
                                          uint32_t* max_indices,
                                          std::size_t batch,
                                          std::size_t channels,
                                          std::size_t height,
                                          std::size_t width) {
    const std::size_t ow = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t oh = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t c = blockIdx.z % channels;
    const std::size_t n = blockIdx.z / channels;

    const std::size_t out_h = height / 2;
    const std::size_t out_w = width / 2;
    if (n >= batch || oh >= out_h || ow >= out_w) return;

    auto idx4 = [&](std::size_t n, std::size_t c, std::size_t h,
                    std::size_t w, std::size_t C, std::size_t H,
                    std::size_t W) {
        return ((n * C + c) * H + h) * W + w;
    };

    const std::size_t h0 = oh * 2;
    const std::size_t w0 = ow * 2;
    float max_val = -1e20f;
    uint32_t max_idx = 0;

    for (std::size_t kh = 0; kh < 2; ++kh) {
        for (std::size_t kw = 0; kw < 2; ++kw) {
            const std::size_t ih = h0 + kh;
            const std::size_t iw = w0 + kw;
            const std::size_t in_id =
                idx4(n, c, ih, iw, channels, height, width);
            float v = input[in_id];
            if (v > max_val) {
                max_val = v;
                max_idx = static_cast<uint32_t>(in_id);
            }
        }
    }
    const std::size_t out_id =
        idx4(n, c, oh, ow, channels, out_h, out_w);
    output[out_id] = max_val;
    if (max_indices) {
        max_indices[out_id] = max_idx;
    }
}

void maxpool2x2_forward_naive(const float* input, float* output,
                              uint32_t* max_indices, std::size_t batch,
                              std::size_t channels, std::size_t height,
                              std::size_t width) {
    const std::size_t out_h = height / 2;
    const std::size_t out_w = width / 2;
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y,
              batch * channels);
    maxpool2x2_forward_kernel<<<grid, block>>>(input, output, max_indices,
                                               batch, channels, height, width);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== UpSample 2x2 forward =====================
__global__ void upsample2x2_forward_kernel(const float* input, float* output,
                                           std::size_t batch,
                                           std::size_t channels,
                                           std::size_t height,
                                           std::size_t width) {
    const std::size_t ow = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t oh = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t c = blockIdx.z % channels;
    const std::size_t n = blockIdx.z / channels;

    const std::size_t out_h = height * 2;
    const std::size_t out_w = width * 2;
    if (n >= batch || oh >= out_h || ow >= out_w) return;

    std::size_t ih = oh / 2;
    std::size_t iw = ow / 2;
    auto idx4 = [&](std::size_t n, std::size_t c, std::size_t h,
                    std::size_t w, std::size_t C, std::size_t H,
                    std::size_t W) {
        return ((n * C + c) * H + h) * W + w;
    };
    output[idx4(n, c, oh, ow, channels, out_h, out_w)] =
        input[idx4(n, c, ih, iw, channels, height, width)];
}

void upsample2x2_forward_naive(const float* input, float* output,
                               std::size_t batch, std::size_t channels,
                               std::size_t height, std::size_t width) {
    const std::size_t out_h = height * 2;
    const std::size_t out_w = width * 2;
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y,
              batch * channels);
    upsample2x2_forward_kernel<<<grid, block>>>(input, output, batch, channels,
                                                height, width);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== MSE loss forward/backward =====================
__global__ void mse_loss_forward_kernel(const float* pred, const float* target,
                                        float* loss, std::size_t total) {
    extern __shared__ float shmem[];
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t tid = threadIdx.x;
    float val = 0.0f;
    if (idx < total) {
        float diff = pred[idx] - target[idx];
        val = diff * diff;
    }
    shmem[tid] = val;
    __syncthreads();

    // reduction in block
    for (std::size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shmem[tid] += shmem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss, shmem[0]);
    }
}

__global__ void mse_loss_divide_kernel(float* loss, std::size_t total) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *loss /= static_cast<float>(total);
    }
}

void mse_loss_forward_naive(const float* pred, const float* target,
                            float* loss, std::size_t total) {
    CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));
    const int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    mse_loss_forward_kernel<<<grid, block, block * sizeof(float)>>>(pred, target,
                                                                    loss, total);
    CUDA_CHECK(cudaGetLastError());
    mse_loss_divide_kernel<<<1, 1>>>(loss, total);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mse_loss_backward_kernel(const float* pred, const float* target,
                                         float* grad, std::size_t total,
                                         float scale) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        grad[idx] = (pred[idx] - target[idx]) * scale;
    }
}

void mse_loss_backward_naive(const float* pred, const float* target,
                             float* grad, std::size_t total) {
    const float scale = 2.0f / static_cast<float>(total);
    const int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    mse_loss_backward_kernel<<<grid, block>>>(pred, target, grad, total, scale);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== Forward pipeline (naive) =====================
void forward_naive(GPUAutoencoder& gpu, std::size_t batch) {
    if (batch > gpu.batch_size_max()) {
        throw std::runtime_error("Batch exceeds GPU batch_size_max");
    }
    // Shapes are fixed by architecture.
    // act1: N,256,32,32
    conv2d_forward_relu(gpu.input_buf(), gpu.w1(), gpu.b1(), gpu.act1(),
                        batch, 3, 256, 32, 32);
    maxpool2x2_forward_naive(gpu.act1(), gpu.act2(), gpu.pool1_indices(), batch, 256, 32, 32);

    conv2d_forward_relu(gpu.act2(), gpu.w2(), gpu.b2(), gpu.act3(),
                        batch, 256, 128, 16, 16);
    maxpool2x2_forward_naive(gpu.act3(), gpu.act4(), gpu.pool2_indices(), batch, 128, 16, 16);

    conv2d_forward_relu(gpu.act4(), gpu.w3(), gpu.b3(), gpu.act5(),
                        batch, 128, 128, 8, 8);
    upsample2x2_forward_naive(gpu.act5(), gpu.act6(), batch, 128, 8, 8);

    conv2d_forward_relu(gpu.act6(), gpu.w4(), gpu.b4(), gpu.act7(),
                        batch, 128, 256, 16, 16);
    upsample2x2_forward_naive(gpu.act7(), gpu.act8(), batch, 256, 16, 16);

    conv2d_forward_naive(gpu.act8(), gpu.w5(), gpu.b5(), gpu.act9(),
                         batch, 256, 3, 32, 32);
    // No activation at output.
}

// ===================== ReLU backward =====================
__global__ void relu_backward_kernel(const float* grad_out, const float* output,
                                     float* grad_in, std::size_t total) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        grad_in[idx] = (output[idx] > 0.0f) ? grad_out[idx] : 0.0f;
    }
}

void relu_backward_naive(const float* grad_out, const float* output,
                         float* grad_in, std::size_t total) {
    const int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    relu_backward_kernel<<<grid, block>>>(grad_out, output, grad_in, total);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== MaxPool 2x2 backward =====================
__global__ void maxpool2x2_backward_kernel(const float* grad_out,
                                           const uint32_t* max_indices,
                                           float* grad_in, std::size_t batch,
                                           std::size_t channels,
                                           std::size_t height,
                                           std::size_t width) {
    const std::size_t ow = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t oh = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t c = blockIdx.z % channels;
    const std::size_t n = blockIdx.z / channels;
    const std::size_t out_h = height / 2;
    const std::size_t out_w = width / 2;
    if (n >= batch || oh >= out_h || ow >= out_w) return;

    auto idx4 = [&](std::size_t n, std::size_t c, std::size_t h,
                    std::size_t w, std::size_t C, std::size_t H,
                    std::size_t W) {
        return ((n * C + c) * H + h) * W + w;
    };
    const std::size_t out_id =
        idx4(n, c, oh, ow, channels, out_h, out_w);
    const uint32_t in_id = max_indices[out_id];
    grad_in[in_id] = grad_out[out_id];  // stride=2, non-overlapping
}

void maxpool2x2_backward_naive(const float* grad_out,
                               const uint32_t* max_indices, float* grad_in,
                               std::size_t batch, std::size_t channels,
                               std::size_t height, std::size_t width) {
    const std::size_t out_h = height / 2;
    const std::size_t out_w = width / 2;
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y,
              batch * channels);
    cudaMemset(grad_in, 0, batch * channels * height * width * sizeof(float));
    maxpool2x2_backward_kernel<<<grid, block>>>(grad_out, max_indices, grad_in,
                                                batch, channels, height, width);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== UpSample 2x2 backward =====================
__global__ void upsample2x2_backward_kernel(const float* grad_out,
                                            float* grad_in,
                                            std::size_t batch,
                                            std::size_t channels,
                                            std::size_t height,
                                            std::size_t width) {
    const std::size_t w_in = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t h_in = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t c = blockIdx.z % channels;
    const std::size_t n = blockIdx.z / channels;
    if (n >= batch || h_in >= height || w_in >= width) return;

    auto idx4 = [&](std::size_t n, std::size_t c, std::size_t h,
                    std::size_t w, std::size_t C, std::size_t H,
                    std::size_t W) {
        return ((n * C + c) * H + h) * W + w;
    };
    const std::size_t out_h = height * 2;
    const std::size_t out_w = width * 2;
    const std::size_t oh = h_in * 2;
    const std::size_t ow = w_in * 2;
    float sum = 0.0f;
    sum += grad_out[idx4(n, c, oh, ow, channels, out_h, out_w)];
    sum += grad_out[idx4(n, c, oh + 1, ow, channels, out_h, out_w)];
    sum += grad_out[idx4(n, c, oh, ow + 1, channels, out_h, out_w)];
    sum += grad_out[idx4(n, c, oh + 1, ow + 1, channels, out_h, out_w)];
    grad_in[idx4(n, c, h_in, w_in, channels, height, width)] = sum;
}

void upsample2x2_backward_naive(const float* grad_out, float* grad_in,
                                std::size_t batch, std::size_t channels,
                                std::size_t height, std::size_t width) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
              batch * channels);
    upsample2x2_backward_kernel<<<grid, block>>>(grad_out, grad_in, batch,
                                                 channels, height, width);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== Conv2D backward =====================
// __global__ void conv2d_backward_kernel(const float* input,
//                                        const float* grad_out,
//                                        const float* weight, float* grad_input,
//                                        float* grad_weight, float* grad_bias,
//                                        std::size_t batch, std::size_t in_c,
//                                        std::size_t out_c, std::size_t height,
//                                        std::size_t width) {
//     const std::size_t w_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const std::size_t h_idx = blockIdx.y * blockDim.y + threadIdx.y;
//     const std::size_t oc = blockIdx.z % out_c;
//     const std::size_t n = blockIdx.z / out_c;
//     if (n >= batch || h_idx >= height || w_idx >= width) return;

//     auto idx4 = [&](std::size_t n, std::size_t c, std::size_t h,
//                     std::size_t w, std::size_t C, std::size_t H,
//                     std::size_t W) {
//         return ((n * C + c) * H + h) * W + w;
//     };
//     auto widx = [&](std::size_t oc, std::size_t ic, std::size_t kh,
//                     std::size_t kw, std::size_t in_c) {
//         return ((oc * in_c + ic) * 3 + kh) * 3 + kw;
//     };

//     const float go = grad_out[idx4(n, oc, h_idx, w_idx, out_c, height, width)];
//     // grad_bias
//     atomicAdd(&grad_bias[oc], go);

//     for (std::size_t ic = 0; ic < in_c; ++ic) {
//         for (std::size_t kh = 0; kh < 3; ++kh) {
//             for (std::size_t kw = 0; kw < 3; ++kw) {
//                 int in_h = static_cast<int>(h_idx) + static_cast<int>(kh) - 1;
//                 int in_w = static_cast<int>(w_idx) + static_cast<int>(kw) - 1;
//                 if (in_h < 0 || in_w < 0 || in_h >= static_cast<int>(height) ||
//                     in_w >= static_cast<int>(width)) {
//                     continue;
//                 }
//                 std::size_t in_id =
//                     idx4(n, ic, static_cast<std::size_t>(in_h),
//                          static_cast<std::size_t>(in_w), in_c, height, width);
//                 std::size_t w_id = widx(oc, ic, kh, kw, in_c);
//                 atomicAdd(&grad_weight[w_id], input[in_id] * go);
//                 atomicAdd(&grad_input[in_id], weight[w_id] * go);
//             }
//         }
//     }
// }

// ===================== Conv2D backward (OPTIMIZED: Tiling & Split) =====================

// Cấu hình Tiling
#define TILE_W 16
#define TILE_H 16
#define R 1 
#define SMEM_W (TILE_W + 2 * R)
#define SMEM_H (TILE_H + 2 * R)

// Kernel 1: Tính Gradient Input (Backward Data) - Dùng Shared Memory Tiling
// Kernel này loại bỏ hoàn toàn atomicAdd khi tính g_act
// ===================== Kernel Backward Data "PRO" (Full Halo Loading) =====================

__global__ void conv2d_backward_data_tiled(
    const float* __restrict__ grad_out, 
    const float* __restrict__ weight,
    float* __restrict__ grad_input,
    int batch, int in_c, int out_c, int height, int width) 
{
    // Kích thước Tile Input: 16x16
    // Kích thước cần Load (kèm viền): 18x18
    __shared__ float s_grad_out[SMEM_H * SMEM_W]; // Dùng mảng 1D cho dễ map linear

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx; // Linear Thread ID (0 -> 255)

    int n = blockIdx.z / in_c;
    int ic = blockIdx.z % in_c;

    // Tọa độ gốc của Block trong ảnh Input (Top-Left của Tile 16x16)
    int block_start_y = blockIdx.y * TILE_H;
    int block_start_x = blockIdx.x * TILE_W;

    if (n >= batch) return;

    float d_input_val = 0.0f;

    // Loop over Output Channels
    for (int oc = 0; oc < out_c; ++oc) {
        
        // --- BƯỚC 1: COOPERATIVE LOADING (Load cả vùng Halo) ---
        // Ta cần load 18x18 = 324 phần tử.
        // Có 256 threads. Thread sẽ load nhiều hơn 1 phần tử.
        
        int num_elements_to_load = SMEM_H * SMEM_W; // 324
        
        for (int i = tid; i < num_elements_to_load; i += blockDim.x * blockDim.y) {
            // Map linear index 'i' sang tọa độ trong Shared Mem (18x18)
            int s_y = i / SMEM_W;
            int s_x = i % SMEM_W;

            // Map sang tọa độ Global (grad_out). 
            // Lưu ý: SMEM[1][1] tương ứng với BlockStart[0][0].
            // Nên SMEM[0][0] tương ứng BlockStart[-1][-1] (padding thật của ảnh)
            int g_y = block_start_y + s_y - R; // trừ R để lấy viền
            int g_x = block_start_x + s_x - R;

            float val = 0.0f;
            if (g_y >= 0 && g_y < height && g_x >= 0 && g_x < width) {
                val = grad_out[((n * out_c + oc) * height + g_y) * width + g_x];
            }
            s_grad_out[i] = val;
        }

        __syncthreads(); // Đợi tất cả load xong

        // --- BƯỚC 2: TÍNH TOÁN (Không cần check if/else nữa) ---
        // Chỉ tính cho 16x16 pixels bên trong
        int iy = block_start_y + ty;
        int ix = block_start_x + tx;

        if (iy < height && ix < width) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    // Logic Convolution Backward
                    // SMEM đã chứa đủ dữ liệu. Pixel (ty, tx) cần dữ liệu xung quanh.
                    // Trong SMEM 18x18, pixel (ty, tx) nằm ở vị trí (ty+R, tx+R).
                    // Khi convolution, ta truy cập lùi: (ty+R - kh + 1, tx+R - kw + 1)
                    // (Lưu ý công thức index phụ thuộc vào việc bạn coi weight là cross-correlation hay conv)
                    // Với backward chuẩn:
                    
                    int s_idx_y = ty + R - kh + 1;
                    int s_idx_x = tx + R - kw + 1;
                    
                    float go_val = s_grad_out[s_idx_y * SMEM_W + s_idx_x];

                    int w_idx = ((oc * in_c + ic) * 3 + kh) * 3 + kw;
                    d_input_val += go_val * weight[w_idx];
                }
            }
        }
        __syncthreads(); // Sync trước khi load channel tiếp theo
    }

    // Ghi kết quả
    int iy = block_start_y + ty;
    int ix = block_start_x + tx;
    if (iy < height && ix < width) {
        int idx = ((n * in_c + ic) * height + iy) * width + ix;
        grad_input[idx] = d_input_val;
    }
}

// Kernel 2: Tính Gradient Weight & Bias (Backward Filter) - Dùng Register Accumulation
// Giảm số lượng atomicAdd từ (H*W) lần xuống 1 lần/thread
__global__ void conv2d_backward_filter_optimized(
    const float* __restrict__ input, 
    const float* __restrict__ grad_out,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int batch, int in_c, int out_c, int height, int width) 
{
    // 1. SHARED MEMORY: Lưu gradient của filter 3x3 và bias
    // Kích thước rất nhỏ: 9 floats cho weights + 1 float cho bias
    __shared__ float s_dw[3][3];
    __shared__ float s_db;

    // Khởi tạo Shared Memory về 0 (Chỉ cần Thread 0 làm)
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid == 0) {
        for(int i=0; i<3; ++i)
            for(int j=0; j<3; ++j)
                s_dw[i][j] = 0.0f;
        s_db = 0.0f;
    }
    __syncthreads(); // Đợi khởi tạo xong

    int oc = blockIdx.z;
    int ic = blockIdx.y;
    int tile_idx = blockIdx.x;
    
    // 2. REGISTER ACCUMULATION (Cấp 1)
    // Mỗi thread tự tính phần của mình vào biến cục bộ (nhanh nhất)
    float r_dw[3][3] = {0}; 
    float r_db = 0.0f;

    int stride = blockDim.x * blockDim.y; 
    int total_pixels = batch * height * width;
    
    // Grid-stride loop duyệt qua ảnh
    for (int i = tid + tile_idx * stride; i < total_pixels; i += stride * gridDim.x) {
        int n = i / (height * width);
        int rem = i % (height * width);
        int r = rem / width;
        int c = rem % width;

        float go = grad_out[((n * out_c + oc) * height + r) * width + c];
        
        r_db += go;

        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int in_r = r + kh - 1; 
                int in_c_pos = c + kw - 1;
                
                if (in_r >= 0 && in_r < height && in_c_pos >= 0 && in_c_pos < width) {
                    float val = input[((n * in_c + ic) * height + in_r) * width + in_c_pos];
                    r_dw[kh][kw] += val * go;
                }
            }
        }
    }

    // 3. SHARED MEMORY REDUCTION (Cấp 2)
    // Các threads cộng dồn biến cục bộ vào shared memory
    // atomicAdd trên Shared Memory rất nhanh (được hỗ trợ phần cứng từ Maxwell trở lên)
    for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
            atomicAdd(&s_dw[kh][kw], r_dw[kh][kw]);
        }
    }
    atomicAdd(&s_db, r_db);

    __syncthreads(); // Đợi tất cả threads cộng xong

    // 4. GLOBAL MEMORY UPDATE (Cấp 3)
    // Chỉ duy nhất Thread 0 đại diện Block ghi ra Global Memory
    if (tid == 0) {
        // Cộng Weight
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int w_idx = ((oc * in_c + ic) * 3 + kh) * 3 + kw;
                atomicAdd(&grad_weight[w_idx], s_dw[kh][kw]);
            }
        }

        // Cộng Bias
        // Logic Bias: Bias chỉ phụ thuộc Out Channel (oc).
        // Nhưng Grid của ta chạy song song theo cả In Channel (ic) và Tile.
        // Để tránh cộng trùng lặp (ví dụ 128 ic block cùng cộng vào 1 bias -> sai),
        // ta quy ước: Chỉ block có ic=0 và tile_idx=0 mới được quyền cập nhật bias.
        if (ic == 0 && tile_idx == 0) {
            atomicAdd(&grad_bias[oc], s_db);
        }
    }
}

void conv2d_backward_naive(const float* input, const float* grad_out,
                           const float* weight, float* grad_input,
                           float* grad_weight, float* grad_bias,
                           std::size_t batch, std::size_t in_c,
                           std::size_t out_c, std::size_t height,
                           std::size_t width) {
    
    // 1. Launch Kernel tính Gradient Data (grad_input)
    dim3 block(16, 16);
    dim3 grid_data((width + block.x - 1) / block.x, 
                   (height + block.y - 1) / block.y, 
                   batch * in_c); 
    
    conv2d_backward_data_tiled<<<grid_data, block>>>(
        grad_out, weight, grad_input, batch, in_c, out_c, height, width);
    CUDA_CHECK(cudaGetLastError());

    // 2. Launch Kernel tính Gradient Weight & Bias (REDUCTION)
    // Tăng số lượng tiles lên để tận dụng nhiều SMs hơn nếu ảnh lớn
    // Với CIFAR 32x32, batch 64, tổng pixel lớn -> tăng tile lên 8 hoặc 16
    int num_tiles = 8; 
    dim3 grid_filter(num_tiles, in_c, out_c);
    dim3 block_filter(256); 
    
    conv2d_backward_filter_optimized<<<grid_filter, block_filter>>>(
        input, grad_out, grad_weight, grad_bias, batch, in_c, out_c, height, width);
    CUDA_CHECK(cudaGetLastError());
}

float GPUAutoencoder::compute_mse_loss(std::size_t batch) {
    // act9 is the output, input_buf is the target (since this is an Autoencoder)
    std::size_t total_elems = batch * 3 * 32 * 32;

    // Call the existing naive kernel
    mse_loss_forward_naive(d_act9_, d_input_, d_loss_val_, total_elems);

    // Copy only the single scalar float back to Host for logging
    float host_loss = 0.0f;
    CUDA_CHECK(cudaMemcpy(&host_loss, d_loss_val_, sizeof(float), cudaMemcpyDeviceToHost));

    return host_loss;
}

void GPUAutoencoder::compute_mse_gradient(std::size_t batch) {
    std::size_t total_elems = batch * 3 * 32 * 32;

    // Compute gradient and write directly to d_g_act9_ on device
    mse_loss_backward_naive(d_act9_, d_input_, d_g_act9_, total_elems);
}

#include "gpu_autoencoder.h"
#include <stdexcept>
#include <vector>
