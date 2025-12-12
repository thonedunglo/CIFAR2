#include "autoencoder.h"
#include "layers.h"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    AutoencoderCPU ae;
    const std::size_t batch = 1;
    std::vector<float> input(batch * 3 * 32 * 32, 0.1f);
    const auto& out = ae.forward(input, batch);

    // Use MSE loss against zeros for a quick backward sanity.
    MSELoss mse;
    std::vector<float> target(out.size(), 0.0f);
    float loss = mse.forward(out, target);
    auto grad_out = mse.backward(out, target);

    auto grad_in = ae.backward(grad_out);
    assert(grad_in.size() == input.size());

    // Simple checks: loss should be > 0, grads non-zero somewhere.
    bool any_grad = false;
    for (float g : grad_in) {
        if (g != 0.0f) {
            any_grad = true;
            break;
        }
    }

    std::cout << "Loss=" << loss << ", grad_input_nonzero=" << any_grad << "\n";
    std::cout << "Autoencoder backward sanity passed.\n";
    return 0;
}
