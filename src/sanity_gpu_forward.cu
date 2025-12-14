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

    // Prepare input on host.
    std::vector<float> h_in(batch * 3 * 32 * 32, 0.1f);
    gpu.copy_input_to_device(h_in, batch);

    // Run forward (naive)
    forward_naive(gpu, batch);

    // Copy output back and check size/non-NaN.
    std::vector<float> h_out(batch * 3 * 32 * 32, 0.0f);
    cudaMemcpy(h_out.data(), gpu.act9(), h_out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Simple check: finite values and not all zero (weights are random).
    bool finite = true;
    bool any_nonzero = false;
    for (float v : h_out) {
        if (!std::isfinite(v)) finite = false;
        if (v != 0.0f) any_nonzero = true;
    }
    assert(finite);
    assert(any_nonzero);
    std::cout << "GPU forward naive sanity passed.\n";
    return 0;
}
