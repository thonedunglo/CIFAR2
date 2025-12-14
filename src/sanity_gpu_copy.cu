#include "gpu_autoencoder.h"

#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    const std::size_t batch = 2;
    const std::size_t elems = batch * 3 * 32 * 32;
    std::vector<float> h_in(elems);
    for (std::size_t i = 0; i < elems; ++i) h_in[i] = static_cast<float>(i);

    GPUAutoencoder gpu(batch);
    gpu.copy_input_to_device(h_in, batch);

    std::vector<float> h_out(elems, 0.0f);
    cudaMemcpy(h_out.data(), gpu.input_buf(), elems * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < elems; ++i) {
        assert(h_out[i] == h_in[i]);
    }
    std::cout << "GPU H2D copy test passed.\n";
    return 0;
}
