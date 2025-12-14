#include "gpu_autoencoder.h"

#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    const std::size_t batch = 1;
    AutoencoderCPU cpu;
    GPUAutoencoder gpu(batch);
    gpu.load_weights(cpu);

    // Prepare dummy grads on device for conv1 weight/bias
    std::vector<float> grad_w(cpu.conv1().weights().size(), 1.0f);
    std::vector<float> grad_b(cpu.conv1().bias().size(), 1.0f);

    cudaMemcpy(gpu.gw1(), grad_w.data(),
               grad_w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.gb1(), grad_b.data(),
               grad_b.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Snapshot weights before
    std::vector<float> w_before = cpu.conv1().weights();
    std::vector<float> b_before = cpu.conv1().bias();

    float lr = 0.01f;
    gpu.step(lr);
    // Copy back weights
    gpu.save_weights(cpu);
    auto w_after = cpu.conv1().weights();
    auto b_after = cpu.conv1().bias();

    for (std::size_t i = 0; i < w_before.size(); ++i) {
        assert(std::abs(w_after[i] - (w_before[i] - lr * 1.0f)) < 1e-5f);
    }
    for (std::size_t i = 0; i < b_before.size(); ++i) {
        assert(std::abs(b_after[i] - (b_before[i] - lr * 1.0f)) < 1e-5f);
    }

    std::cout << "GPU SGD update test passed.\n";
    return 0;
}
