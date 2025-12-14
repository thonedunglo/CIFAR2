#pragma once

#include <array>
#include <cstddef>
#include <random>

#include "layers.h"

// CPU Autoencoder wrapper: holds architecture definition and weight init.
// Forward/backward will be added in subsequent subtasks.
class AutoencoderCPU {
public:
    AutoencoderCPU();

    // Reinitialize weights/bias with He/Kaiming (fan_in) and bias=0.
    void init_weights(unsigned int seed = std::random_device{}());

    // Forward pass encoder->decoder. Returns reference to internal output buffer.
    const std::vector<float>& forward(const std::vector<float>& input,
                                      std::size_t batch);

    // Encoder only: returns reference to latent buffer (after pool2).
    const std::vector<float>& encode(const std::vector<float>& input,
                                     std::size_t batch);

    // Extract features (latent) for a batch. Returns a copy sized
    // batch x 128 x 8 x 8 (8192 dims per sample).
    std::vector<float> extract_features(const std::vector<float>& input,
                                        std::size_t batch);

    // Backward given grad_output (same shape as forward output). Accumulates
    // gradients into conv layers and returns grad_input w.r.t original input.
    std::vector<float> backward(const std::vector<float>& grad_output);

    // Zero gradients of all conv layers.
    void zero_grad();

    // SGD update all conv layers.
    void step(float lr);

    // Save/load weights to/from binary file.
    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);

    // Accessors to conv layers (for later forward/backward and testing).
    Conv2D& conv1() { return conv1_; }
    Conv2D& conv2() { return conv2_; }
    Conv2D& conv3() { return conv3_; }
    Conv2D& conv4() { return conv4_; }
    Conv2D& conv5() { return conv5_; }
    const Conv2D& conv1() const { return conv1_; }
    const Conv2D& conv2() const { return conv2_; }
    const Conv2D& conv3() const { return conv3_; }
    const Conv2D& conv4() const { return conv4_; }
    const Conv2D& conv5() const { return conv5_; }

private:
    void init_conv_weights(Conv2D& conv, float scale, std::mt19937& rng);

    // Encoder
    Conv2D conv1_;  // 3 -> 256
    ReLU relu1_;
    MaxPool2x2 pool1_;

    Conv2D conv2_;  // 256 -> 128
    ReLU relu2_;
    MaxPool2x2 pool2_;

    // Bottleneck / decoder
    Conv2D conv3_;  // 128 -> 128
    ReLU relu3_;
    UpSample2x2 up1_;

    Conv2D conv4_;  // 128 -> 256
    ReLU relu4_;
    UpSample2x2 up2_;

    Conv2D conv5_;  // 256 -> 3 (no activation here)

    // Activations cache for forward/backward.
    std::vector<float> act1_;  // after conv1+relu1, shape: N,256,32,32
    std::vector<float> act2_;  // after pool1, shape: N,256,16,16
    std::vector<float> act3_;  // after conv2+relu2, shape: N,128,16,16
    std::vector<float> act4_;  // after pool2 (latent), shape: N,128,8,8
    std::vector<float> act5_;  // after conv3+relu3, shape: N,128,8,8
    std::vector<float> act6_;  // after up1, shape: N,128,16,16
    std::vector<float> act7_;  // after conv4+relu4, shape: N,256,16,16
    std::vector<float> act8_;  // after up2, shape: N,256,32,32
    std::vector<float> act9_;  // after conv5 (output), shape: N,3,32,32

    std::size_t cached_batch_ = 0;
};
