#include "autoencoder.h"
#include "layers.h"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    AutoencoderCPU ae;
    const std::size_t batch = 1;
    std::vector<float> input(batch * 3 * 32 * 32, 0.5f);
    std::vector<float> target(input.size(), 0.0f);
    MSELoss mse;

    // Forward
    const auto& out1 = ae.forward(input, batch);
    float loss1 = mse.forward(out1, target);
    auto grad_out = mse.backward(out1, target);

    // Backward + update
    ae.backward(grad_out);
    ae.step(1e-3f);

    // Forward again after update to see if loss decreases (not guaranteed but often yes with small init).
    const auto& out2 = ae.forward(input, batch);
    float loss2 = mse.forward(out2, target);

    std::cout << "Loss before update: " << loss1 << "\n";
    std::cout << "Loss after  update: " << loss2 << "\n";
    std::cout << "SGD sanity finished (loss may or may not decrease on random init).\n";
    return 0;
}
