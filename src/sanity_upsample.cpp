#include "layers.h"

#include <cassert>
#include <iostream>
#include <vector>

bool nearly_equal(float a, float b, float eps = 1e-6f) {
    return std::abs(a - b) <= eps;
}

int main() {
    UpSample2x2 up;

    {
        // Forward test: 1x1x2x2
        // Input:
        // 1 2
        // 3 4
        std::vector<float> input = {1, 2, 3, 4};
        auto out = up.forward(input, 1, 1, 2, 2);
        // Expected output 4x4 block with nearest neighbor repeats.
        const std::vector<float> expected_out = {
            1, 1, 2, 2,
            1, 1, 2, 2,
            3, 3, 4, 4,
            3, 3, 4, 4,
        };
        assert(out.size() == expected_out.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            assert(nearly_equal(out[i], expected_out[i]));
        }
        std::cout << "UpSample forward test: OK\n";
    }

    {
        // Backward test: grad_output all ones on 4x4 should sum to 4 at each input.
        std::vector<float> input = {0, 0, 0, 0};  // not used in backward
        up.forward(input, 1, 1, 2, 2);  // set cache

        std::vector<float> grad_out(16, 1.0f);
        auto grad_in = up.backward(grad_out);
        const std::vector<float> expected_grad_in = {4, 4, 4, 4};
        for (std::size_t i = 0; i < grad_in.size(); ++i) {
            assert(nearly_equal(grad_in[i], expected_grad_in[i]));
        }
        std::cout << "UpSample backward test: OK\n";
    }

    std::cout << "All UpSample2x2 sanity tests passed.\n";
    return 0;
}
