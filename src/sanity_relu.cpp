#include "layers.h"

#include <cassert>
#include <iostream>
#include <vector>

bool nearly_equal(float a, float b, float eps = 1e-6f) {
    return std::abs(a - b) <= eps;
}

int main() {
    ReLU relu;
    const std::vector<float> input = {-1.0f, 0.0f, 0.5f, 2.0f, -3.0f};
    auto out = relu.forward(input);

    const std::vector<float> expected_out = {0.0f, 0.0f, 0.5f, 2.0f, 0.0f};
    for (std::size_t i = 0; i < out.size(); ++i) {
        assert(nearly_equal(out[i], expected_out[i]));
    }

    const std::vector<float> grad_out = {1.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    auto grad_in = relu.backward(grad_out);
    const std::vector<float> expected_grad_in = {0.0f, 0.0f, 2.0f, 3.0f, 0.0f};
    for (std::size_t i = 0; i < grad_in.size(); ++i) {
        assert(nearly_equal(grad_in[i], expected_grad_in[i]));
    }

    std::cout << "ReLU sanity test passed.\n";
    return 0;
}
