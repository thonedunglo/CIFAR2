#include "gpu_autoencoder.h"

#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    std::vector<float> pred = {1.0f, 2.0f};
    std::vector<float> target = {0.0f, 0.0f};
    const std::size_t total = pred.size();

    float *d_pred = nullptr, *d_target = nullptr, *d_loss = nullptr, *d_grad = nullptr;
    cudaMalloc(&d_pred, total * sizeof(float));
    cudaMalloc(&d_target, total * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));
    cudaMalloc(&d_grad, total * sizeof(float));

    cudaMemcpy(d_pred, pred.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    mse_loss_forward_naive(d_pred, d_target, d_loss, total);
    float h_loss = 0.0f;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    // Expected loss = ((1)^2 + (2)^2)/2 = 2.5
    assert(std::abs(h_loss - 2.5f) < 1e-5f);

    mse_loss_backward_naive(d_pred, d_target, d_grad, total);
    std::vector<float> h_grad(total, 0.0f);
    cudaMemcpy(h_grad.data(), d_grad, total * sizeof(float), cudaMemcpyDeviceToHost);
    // Expected grad = [1, 2]
    assert(std::abs(h_grad[0] - 1.0f) < 1e-5f);
    assert(std::abs(h_grad[1] - 2.0f) < 1e-5f);

    std::cout << "GPU MSE forward/backward test passed.\n";

    cudaFree(d_pred);
    cudaFree(d_target);
    cudaFree(d_loss);
    cudaFree(d_grad);
    return 0;
}
