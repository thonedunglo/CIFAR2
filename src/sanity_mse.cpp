#include "layers.h"

#include <cassert>
#include <iostream>
#include <vector>

bool nearly_equal(float a, float b, float eps = 1e-6f) {
    return std::abs(a - b) <= eps;
}

int main() {
    MSELoss mse;

    {
        std::vector<float> pred = {1.0f, 2.0f};
        std::vector<float> target = {0.0f, 0.0f};
        float loss = mse.forward(pred, target);
        // ((1)^2 + (2)^2) / 2 = 2.5
        assert(nearly_equal(loss, 2.5f));
        auto grad = mse.backward(pred, target);
        std::vector<float> expected_grad = {1.0f, 2.0f};
        for (std::size_t i = 0; i < grad.size(); ++i) {
            assert(nearly_equal(grad[i], expected_grad[i]));
        }
        std::cout << "MSE test 1 passed.\n";
    }

    {
        // pred == target -> loss 0, grad 0
        std::vector<float> pred = {0.5f, -1.2f, 3.0f};
        std::vector<float> target = {0.5f, -1.2f, 3.0f};
        float loss = mse.forward(pred, target);
        assert(nearly_equal(loss, 0.0f));
        auto grad = mse.backward(pred, target);
        for (float g : grad) {
            assert(nearly_equal(g, 0.0f));
        }
        std::cout << "MSE test 2 passed.\n";
    }

    std::cout << "All MSE sanity tests passed.\n";
    return 0;
}
