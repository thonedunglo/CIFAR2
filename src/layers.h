#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Simple 3x3 convolution with padding=1, stride=1, channel-first (N, C, H, W).
// Stores weights, biases, and gradients for CPU baseline.
class Conv2D {
public:
    Conv2D(std::size_t in_channels, std::size_t out_channels);

    // Forward: input -> output. Caches input for backward.
    // input shape: N x in_channels x H x W
    // output shape: N x out_channels x H x W
    std::vector<float> forward(const std::vector<float>& input,
                               std::size_t batch,
                               std::size_t height,
                               std::size_t width);

    // Backward: given grad_output (same shape as output), compute grad_input
    // and accumulate grad_weight/grad_bias.
    std::vector<float> backward(const std::vector<float>& grad_output,
                                std::size_t batch,
                                std::size_t height,
                                std::size_t width);

    void zero_grad();
    void step(float lr);

    std::vector<float>& weights() { return weights_; }
    std::vector<float>& bias() { return bias_; }
    const std::vector<float>& weights() const { return weights_; }
    const std::vector<float>& bias() const { return bias_; }
    const std::vector<float>& grad_weights() const { return grad_weights_; }
    const std::vector<float>& grad_bias() const { return grad_bias_; }

private:
    std::size_t in_channels_;
    std::size_t out_channels_;
    std::vector<float> weights_;      // [out_c, in_c, 3, 3]
    std::vector<float> bias_;         // [out_c]
    std::vector<float> grad_weights_; // same shape as weights
    std::vector<float> grad_bias_;    // [out_c]

    // Cache input for backward.
    std::vector<float> input_cache_;
    std::size_t cached_batch_ = 0;
    std::size_t cached_h_ = 0;
    std::size_t cached_w_ = 0;
};

// ReLU activation (in-place forward, cached mask for backward).
class ReLU {
public:
    // Forward: y = max(0, x). Stores mask to reuse in backward.
    std::vector<float> forward(const std::vector<float>& input);

    // Backward: grad_input = grad_output * mask.
    std::vector<float> backward(const std::vector<float>& grad_output);

private:
    std::vector<uint8_t> mask_;  // 1 if input > 0, else 0
};

// MaxPooling 2x2 stride 2 (channel-first). Downsamples H/W by 2.
class MaxPool2x2 {
public:
    // input: N x C x H x W, output: N x C x (H/2) x (W/2)
    std::vector<float> forward(const std::vector<float>& input,
                               std::size_t batch,
                               std::size_t channels,
                               std::size_t height,
                               std::size_t width);

    // grad_output: same shape as forward output. Returns grad_input shape of input.
    std::vector<float> backward(const std::vector<float>& grad_output);

private:
    // Cache for backward
    std::vector<std::size_t> max_indices_;  // position in input for each output
    std::size_t cached_batch_ = 0;
    std::size_t cached_channels_ = 0;
    std::size_t cached_in_h_ = 0;
    std::size_t cached_in_w_ = 0;
};

// Nearest-neighbor upsampling by factor 2 (channel-first).
class UpSample2x2 {
public:
    // input: N x C x H x W, output: N x C x (2H) x (2W)
    std::vector<float> forward(const std::vector<float>& input,
                               std::size_t batch,
                               std::size_t channels,
                               std::size_t height,
                               std::size_t width);

    // grad_output: same shape as forward output. Returns grad_input shape of input.
    std::vector<float> backward(const std::vector<float>& grad_output);

private:
    std::size_t cached_batch_ = 0;
    std::size_t cached_channels_ = 0;
    std::size_t cached_in_h_ = 0;
    std::size_t cached_in_w_ = 0;
};

// Mean Squared Error loss (scalar).
class MSELoss {
public:
    // Returns scalar loss = mean((pred - target)^2).
    float forward(const std::vector<float>& pred,
                  const std::vector<float>& target);

    // Returns grad w.r.t pred: 2*(pred - target)/N, where N = num elements.
    std::vector<float> backward(const std::vector<float>& pred,
                                const std::vector<float>& target) const;
};
