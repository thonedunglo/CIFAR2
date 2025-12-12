#include "layers.h"

#include <cassert>
#include <iostream>
#include <vector>

bool nearly_equal(float a, float b, float eps = 1e-6f) {
    return std::abs(a - b) <= eps;
}

int main() {
    MaxPool2x2 pool;

    {
        // Forward test: simple 1x1x2x2
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
        auto out = pool.forward(input, 1, 1, 2, 2);
        assert(out.size() == 1);
        assert(nearly_equal(out[0], 4.0f));

        std::vector<float> grad_out = {1.0f};
        auto grad_in = pool.backward(grad_out);
        // Only the max position (last element) should receive grad.
        std::vector<float> expected_grad_in = {0.0f, 0.0f, 0.0f, 1.0f};
        for (std::size_t i = 0; i < grad_in.size(); ++i) {
            assert(nearly_equal(grad_in[i], expected_grad_in[i]));
        }
        std::cout << "MaxPool 2x2 simple test: OK\n";
    }

    {
        // Forward/Backward test on 1x1x4x4 with distinct values.
        // Input:
        // 1 2 3 4
        // 5 6 7 8
        // 9 1 2 3
        // 4 5 6 7
        std::vector<float> input = {
            1, 2, 3, 4,  //
            5, 6, 7, 8,  //
            9, 1, 2, 3,  //
            4, 5, 6, 7   //
        };
        auto out = pool.forward(input, 1, 1, 4, 4);
        // Expected pooled (2x2): [6,8; 9,7]
        const std::vector<float> expected_out = {6, 8, 9, 7};
        for (std::size_t i = 0; i < out.size(); ++i) {
            assert(nearly_equal(out[i], expected_out[i]));
        }

        // Backward: grad_out all ones -> grad_in at max positions =1, others 0.
        std::vector<float> grad_out(out.size(), 1.0f);
        auto grad_in = pool.backward(grad_out);
        const std::vector<float> expected_grad_in = {
            0, 0, 0, 0,  //
            0, 1, 0, 1,  //
            1, 0, 0, 0,  //
            0, 0, 0, 1   //
        };
        for (std::size_t i = 0; i < grad_in.size(); ++i) {
            assert(nearly_equal(grad_in[i], expected_grad_in[i]));
        }
        std::cout << "MaxPool 4x4 test: OK\n";
    }

    std::cout << "All MaxPool2x2 sanity tests passed.\n";
    return 0;
}
