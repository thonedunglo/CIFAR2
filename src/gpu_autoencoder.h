#pragma once

#include <cstddef>
#include <vector>

#include "autoencoder.h"

// Simple GPU container for weights/bias/activations. Kernels are added later.
class GPUAutoencoder {
public:
    explicit GPUAutoencoder(std::size_t batch_size_max);
    ~GPUAutoencoder();

    // Copy weights/bias from CPU model to device.
    void load_weights(const AutoencoderCPU& cpu);
    // Copy weights/bias from device back to CPU model.
    void save_weights(AutoencoderCPU& cpu) const;

    // Zero gradients of conv weights/bias on device.
    void zero_grads();

    // Copy input batch (host to device). batch must be <= batch_size_max_.
    void copy_input_to_device(const std::vector<float>& host,
                              std::size_t batch);

    // SGD update for all conv weights/bias on device.
    void step(float lr);

    std::size_t batch_size_max() const { return batch_size_max_; }

    // Device pointers (public getters for later kernel launches).
    float* w1() const { return d_w1_; }
    float* w2() const { return d_w2_; }
    float* w3() const { return d_w3_; }
    float* w4() const { return d_w4_; }
    float* w5() const { return d_w5_; }
    float* b1() const { return d_b1_; }
    float* b2() const { return d_b2_; }
    float* b3() const { return d_b3_; }
    float* b4() const { return d_b4_; }
    float* b5() const { return d_b5_; }

    float* gw1() const { return d_gw1_; }
    float* gw2() const { return d_gw2_; }
    float* gw3() const { return d_gw3_; }
    float* gw4() const { return d_gw4_; }
    float* gw5() const { return d_gw5_; }
    float* gb1() const { return d_gb1_; }
    float* gb2() const { return d_gb2_; }
    float* gb3() const { return d_gb3_; }
    float* gb4() const { return d_gb4_; }
    float* gb5() const { return d_gb5_; }

    // Forward activations buffers.
    float* act1() const { return d_act1_; }
    float* act2() const { return d_act2_; }
    float* act3() const { return d_act3_; }
    float* act4() const { return d_act4_; }
    float* act5() const { return d_act5_; }
    float* act6() const { return d_act6_; }
    float* act7() const { return d_act7_; }
    float* act8() const { return d_act8_; }
    float* act9() const { return d_act9_; }

    // Gradient activations buffers (same shapes as forward outputs).
    float* g_act1() const { return d_g_act1_; }
    float* g_act2() const { return d_g_act2_; }
    float* g_act3() const { return d_g_act3_; }
    float* g_act4() const { return d_g_act4_; }
    float* g_act5() const { return d_g_act5_; }
    float* g_act6() const { return d_g_act6_; }
    float* g_act7() const { return d_g_act7_; }
    float* g_act8() const { return d_g_act8_; }
    float* g_act9() const { return d_g_act9_; }
    float* g_input() const { return d_g_input_; }
    float* input_buf() const { return d_input_; }
    uint32_t* pool1_indices() const { return d_pool1_idx_; }
    uint32_t* pool2_indices() const { return d_pool2_idx_; }

private:
    void alloc_weights_and_grads();
    void alloc_activations(std::size_t batch_size_max);
    void free_all();

    std::size_t batch_size_max_;

    // Weights/bias + grads
    float *d_w1_{}, *d_w2_{}, *d_w3_{}, *d_w4_{}, *d_w5_{};
    float *d_b1_{}, *d_b2_{}, *d_b3_{}, *d_b4_{}, *d_b5_{};
    float *d_gw1_{}, *d_gw2_{}, *d_gw3_{}, *d_gw4_{}, *d_gw5_{};
    float *d_gb1_{}, *d_gb2_{}, *d_gb3_{}, *d_gb4_{}, *d_gb5_{};

    // Activations
    float *d_act1_{}, *d_act2_{}, *d_act3_{}, *d_act4_{}, *d_act5_{},
          *d_act6_{}, *d_act7_{}, *d_act8_{}, *d_act9_{};
    // Grad activations
    float *d_g_act1_{}, *d_g_act2_{}, *d_g_act3_{}, *d_g_act4_{},
          *d_g_act5_{}, *d_g_act6_{}, *d_g_act7_{}, *d_g_act8_{}, *d_g_act9_{};
    float* d_g_input_{};  // grad w.r.t input
    float* d_input_{};    // input buffer (device)
    // MaxPool indices
    uint32_t *d_pool1_idx_{}, *d_pool2_idx_{};

    // Sizes (in elements) for weight/bias/activations.
    std::size_t w1_size_{}, w2_size_{}, w3_size_{}, w4_size_{}, w5_size_{};
    std::size_t b1_size_{}, b2_size_{}, b3_size_{}, b4_size_{}, b5_size_{};
    std::size_t act1_size_{}, act2_size_{}, act3_size_{}, act4_size_{},
        act5_size_{}, act6_size_{}, act7_size_{}, act8_size_{}, act9_size_{};
    std::size_t input_size_{};
};

// Naive convolution forward kernel launcher (padding=1, stride=1, channel-first).
void conv2d_forward_naive(const float* input, const float* weight,
                          const float* bias, float* output, std::size_t batch,
                          std::size_t in_c, std::size_t out_c,
                          std::size_t height, std::size_t width);

// ReLU forward (elementwise). If output == input, works in-place.
void relu_forward_naive(const float* input, float* output, std::size_t total);

// MaxPool 2x2 forward. Stores argmax indices (into input) for backward.
void maxpool2x2_forward_naive(const float* input, float* output,
                              uint32_t* max_indices, std::size_t batch,
                              std::size_t channels, std::size_t height,
                              std::size_t width);

// UpSample 2x2 forward (nearest). output H/W = input H/W * 2.
void upsample2x2_forward_naive(const float* input, float* output,
                               std::size_t batch, std::size_t channels,
                               std::size_t height, std::size_t width);

// MSE loss forward: writes scalar loss to device pointer (mean).
void mse_loss_forward_naive(const float* pred, const float* target,
                            float* loss, std::size_t total);

// MSE loss backward: grad = 2*(pred - target)/total (elementwise).
void mse_loss_backward_naive(const float* pred, const float* target,
                             float* grad, std::size_t total);

// Forward pass (encoder->decoder) using naive kernels. Batch must be
// <= batch_size_max. Input assumed already copied to d_input_.
// Returns device pointer to output (d_act9_) and optionally computes loss.
class GPUAutoencoder;
void forward_naive(GPUAutoencoder& gpu, std::size_t batch);

// Backward kernels (naive):
// ReLU backward: grad_in = (out>0) ? grad_out : 0 (uses output from forward).
void relu_backward_naive(const float* grad_out, const float* output,
                         float* grad_in, std::size_t total);

// MaxPool 2x2 backward: uses max_indices from forward.
void maxpool2x2_backward_naive(const float* grad_out, const uint32_t* max_indices,
                               float* grad_in, std::size_t batch,
                               std::size_t channels, std::size_t height,
                               std::size_t width);

// UpSample 2x2 backward: grad_out shape (2H,2W), grad_in shape (H,W).
void upsample2x2_backward_naive(const float* grad_out, float* grad_in,
                                std::size_t batch, std::size_t channels,
                                std::size_t height, std::size_t width);

// Conv backward: computes grad_input, accumulates grad_weight and grad_bias.
void conv2d_backward_naive(const float* input, const float* grad_out,
                           const float* weight, float* grad_input,
                           float* grad_weight, float* grad_bias,
                           std::size_t batch, std::size_t in_c,
                           std::size_t out_c, std::size_t height,
                           std::size_t width);
