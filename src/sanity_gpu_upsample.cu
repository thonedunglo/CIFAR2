#include "gpu_autoencoder.h"

#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    // Input 1x1x2x2: [[1,2],[3,4]]
    std::vector<float> h_in = {1, 2, 3, 4};
    std::vector<float> h_out(16, 0.0f);
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, h_in.size() * sizeof(float));
    cudaMalloc(&d_out, h_out.size() * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    upsample2x2_forward_naive(d_in, d_out, 1, 1, 2, 2);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::vector<float> expected = {
        1, 1, 2, 2,
        1, 1, 2, 2,
        3, 3, 4, 4,
        3, 3, 4, 4};
    for (std::size_t i = 0; i < h_out.size(); ++i) {
        assert(h_out[i] == expected[i]);
    }
    std::cout << "GPU Upsample forward test passed.\n";

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
