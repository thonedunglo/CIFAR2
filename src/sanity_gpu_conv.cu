#include "autoencoder.h"
#include "gpu_autoencoder.h"

#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    // Simple test: batch=1, in_c=1, out_c=1, H=W=4, kernel center=1 -> output=input.
    const std::size_t batch = 1, in_c = 1, out_c = 1, H = 4, W = 4;
    std::vector<float> h_input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 1, 2, 3,
        4, 5, 6, 7};
    std::vector<float> h_weight(3 * 3, 0.0f);
    h_weight[4] = 1.0f;  // center
    std::vector<float> h_bias(1, 0.0f);
    std::vector<float> h_output(H * W, 0.0f);

    float *d_input = nullptr, *d_weight = nullptr, *d_bias = nullptr,
          *d_output = nullptr;
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_weight, h_weight.size() * sizeof(float));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(float));
    cudaMalloc(&d_output, h_output.size() * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    conv2d_forward_naive(d_input, d_weight, d_bias, d_output, batch, in_c,
                         out_c, H, W);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < h_output.size(); ++i) {
        assert(h_output[i] == h_input[i]);
    }
    std::cout << "GPU conv forward identity test passed.\n";

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
    return 0;
}
