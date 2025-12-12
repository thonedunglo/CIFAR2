#include "autoencoder.h"
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

    // Host input/target
    std::vector<float> h_in(batch * 3 * 32 * 32, 0.1f);
    gpu.copy_input_to_device(h_in, batch);

    // Forward
    forward_naive(gpu, batch);

    // Copy output D2H
    std::vector<float> h_out(h_in.size(), 0.0f);
    cudaMemcpy(h_out.data(), gpu.act9(), h_out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Compute loss on host
    MSELoss mse;
    float loss = mse.forward(h_out, h_in);
    auto grad = mse.backward(h_out, h_in);

    // Check grad size and loss finite
    assert(grad.size() == h_in.size());
    assert(std::isfinite(loss));
    std::cout << "GPU D2H + host loss sanity passed. Loss=" << loss << "\n";
    return 0;
}
