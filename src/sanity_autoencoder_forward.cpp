#include "autoencoder.h"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    AutoencoderCPU ae;
    const std::size_t batch = 2;
    // Dummy input N x 3 x 32 x 32
    std::vector<float> input(batch * 3 * 32 * 32, 0.1f);

    const auto& out = ae.forward(input, batch);
    assert(out.size() == batch * 3 * 32 * 32);
    const auto& latent = ae.encode(input, batch);
    assert(latent.size() == batch * 128 * 8 * 8);

    std::cout << "Forward shapes OK. Output size=" << out.size()
              << ", latent size=" << latent.size() << "\n";
    return 0;
}
