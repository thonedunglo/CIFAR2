#include "gpu_autoencoder.h"

#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    std::vector<float> h_in = {-1.0f, 0.0f, 1.5f, -2.3f, 4.0f};
    std::vector<float> h_out(h_in.size(), 0.0f);
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, h_in.size() * sizeof(float));
    cudaMalloc(&d_out, h_out.size() * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    relu_forward_naive(d_in, d_out, h_in.size());
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::vector<float> expected = {0.0f, 0.0f, 1.5f, 0.0f, 4.0f};
    for (std::size_t i = 0; i < h_in.size(); ++i) {
        assert(h_out[i] == expected[i]);
    }
    std::cout << "GPU ReLU forward test passed.\n";

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
