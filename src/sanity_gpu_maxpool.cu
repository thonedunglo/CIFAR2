#include "gpu_autoencoder.h"

#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    // Input 1x1x4x4
    std::vector<float> h_in = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 1, 2, 3,
        4, 5, 6, 7};
    std::vector<float> h_out(4, 0.0f);
    std::vector<uint32_t> h_idx(4, 0);

    float *d_in = nullptr, *d_out = nullptr;
    uint32_t* d_idx = nullptr;
    cudaMalloc(&d_in, h_in.size() * sizeof(float));
    cudaMalloc(&d_out, h_out.size() * sizeof(float));
    cudaMalloc(&d_idx, h_idx.size() * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    maxpool2x2_forward_naive(d_in, d_out, d_idx, 1, 1, 4, 4);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(), d_idx, h_idx.size() * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    // Expected pooled (2x2): [6,8; 9,7]
    std::vector<float> expected = {6, 8, 9, 7};
    for (std::size_t i = 0; i < h_out.size(); ++i) {
        assert(h_out[i] == expected[i]);
    }
    std::cout << "GPU MaxPool forward test passed.\n";

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_idx);
    return 0;
}
