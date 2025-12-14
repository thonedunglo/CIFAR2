#include "gpu_autoencoder.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    std::vector<float> grad_out = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out = {-1.0f, 0.0f, 0.5f, 2.0f};
    std::vector<float> grad_in(grad_out.size(), 0.0f);
    float *d_go = nullptr, *d_out = nullptr, *d_gi = nullptr;
    cudaMalloc(&d_go, grad_out.size() * sizeof(float));
    cudaMalloc(&d_out, out.size() * sizeof(float));
    cudaMalloc(&d_gi, grad_in.size() * sizeof(float));
    cudaMemcpy(d_go, grad_out.data(), grad_out.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out.data(), out.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    relu_backward_naive(d_go, d_out, d_gi, grad_out.size());
    cudaDeviceSynchronize();

    cudaMemcpy(grad_in.data(), d_gi, grad_in.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    std::vector<float> expected = {0.0f, 0.0f, 3.0f, 4.0f};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        assert(std::abs(grad_in[i] - expected[i]) < 1e-5f);
    }
    std::cout << "GPU ReLU backward test passed.\n";

    cudaFree(d_go);
    cudaFree(d_out);
    cudaFree(d_gi);
    return 0;
}
