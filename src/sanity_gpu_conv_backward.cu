#include "gpu_autoencoder.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// CPU reference for grad_weight/bias with padding=1, stride=1, in_c=out_c=1.
void conv_backward_cpu(const std::vector<float>& input,
                       const std::vector<float>& grad_out, std::size_t H,
                       std::size_t W, std::vector<float>& grad_weight,
                       float& grad_bias) {
    grad_weight.assign(9, 0.0f);
    grad_bias = 0.0f;
    for (std::size_t h = 0; h < H; ++h) {
        for (std::size_t w = 0; w < W; ++w) {
            const float go = grad_out[h * W + w];
            grad_bias += go;
            for (std::size_t kh = 0; kh < 3; ++kh) {
                for (std::size_t kw = 0; kw < 3; ++kw) {
                    int in_h = static_cast<int>(h) + static_cast<int>(kh) - 1;
                    int in_w = static_cast<int>(w) + static_cast<int>(kw) - 1;
                    if (in_h < 0 || in_w < 0 || in_h >= static_cast<int>(H) ||
                        in_w >= static_cast<int>(W))
                        continue;
                    grad_weight[kh * 3 + kw] +=
                        input[in_h * W + in_w] * go;
                }
            }
        }
    }
}

int main() {
    const std::size_t H = 2, W = 2;
    std::vector<float> input = {1, 2, 3, 4};
    std::vector<float> grad_out = {1, 1, 1, 1};

    std::vector<float> grad_w_ref;
    float grad_b_ref = 0.0f;
    conv_backward_cpu(input, grad_out, H, W, grad_w_ref, grad_b_ref);

    float *d_in = nullptr, *d_go = nullptr, *d_w = nullptr, *d_gw = nullptr,
          *d_gb = nullptr, *d_gi = nullptr;
    cudaMalloc(&d_in, input.size() * sizeof(float));
    cudaMalloc(&d_go, grad_out.size() * sizeof(float));
    cudaMalloc(&d_w, 9 * sizeof(float));
    cudaMalloc(&d_gw, 9 * sizeof(float));
    cudaMalloc(&d_gb, sizeof(float));
    cudaMalloc(&d_gi, input.size() * sizeof(float));

    cudaMemcpy(d_in, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_go, grad_out.data(), grad_out.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(d_w, 0, 9 * sizeof(float));
    cudaMemset(d_gw, 0, 9 * sizeof(float));
    cudaMemset(d_gb, 0, sizeof(float));
    cudaMemset(d_gi, 0, input.size() * sizeof(float));

    conv2d_backward_naive(d_in, d_go, d_w, d_gi, d_gw, d_gb, 1, 1, 1, H, W);
    cudaDeviceSynchronize();

    std::vector<float> grad_w(9, 0.0f);
    float grad_b = 0.0f;
    cudaMemcpy(grad_w.data(), d_gw, 9 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&grad_b, d_gb, sizeof(float), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < grad_w.size(); ++i) {
        assert(std::abs(grad_w[i] - grad_w_ref[i]) < 1e-4f);
    }
    assert(std::abs(grad_b - grad_b_ref) < 1e-4f);
    std::cout << "GPU Conv backward grad_weight/bias test passed.\n";

    cudaFree(d_in);
    cudaFree(d_go);
    cudaFree(d_w);
    cudaFree(d_gw);
    cudaFree(d_gb);
    cudaFree(d_gi);
    return 0;
}
