#include "gpu_autoencoder.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    // grad_out for 1x1x4x4 all ones -> grad_in (2x2) should be 4 each.
    std::vector<float> grad_out(16, 1.0f);
    std::vector<float> grad_in(4, 0.0f);

    float *d_go = nullptr, *d_gi = nullptr;
    cudaMalloc(&d_go, grad_out.size() * sizeof(float));
    cudaMalloc(&d_gi, grad_in.size() * sizeof(float));
    cudaMemcpy(d_go, grad_out.data(), grad_out.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    upsample2x2_backward_naive(d_go, d_gi, 1, 1, 2, 2);
    cudaDeviceSynchronize();

    cudaMemcpy(grad_in.data(), d_gi, grad_in.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (float v : grad_in) {
        assert(std::abs(v - 4.0f) < 1e-5f);
    }
    std::cout << "GPU UpSample backward test passed.\n";

    cudaFree(d_go);
    cudaFree(d_gi);
    return 0;
}
