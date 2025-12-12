#include "layers.h"

#include <cassert>
#include <iostream>
#include <vector>

// Helper to compare floats with tolerance.
bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) <= eps;
}

int main() {
    {
        // Test 1: identity-like kernel (center=1, others=0) with pad=1 should
        // reproduce input.
        Conv2D conv(1, 1);
        auto& w = conv.weights();
        std::fill(w.begin(), w.end(), 0.0f);
        w[4] = 1.0f;  // center of 3x3
        conv.bias()[0] = 0.0f;

        const std::vector<float> input = {
            1, 2, 3, 4,  //
            5, 6, 7, 8,  //
            9, 1, 2, 3,  //
            4, 5, 6, 7   //
        };
        auto out = conv.forward(input, /*batch=*/1, /*H=*/4, /*W=*/4);
        assert(out.size() == input.size());
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (!nearly_equal(out[i], input[i])) {
                std::cerr << "Identity conv mismatch at " << i << ": got "
                          << out[i] << " expected " << input[i] << "\n";
                return 1;
            }
        }
        std::cout << "Conv identity test: OK\n";
    }

    {
        // Test 2: all-ones kernel, bias=0, input all-ones.
        // With zero padding, expected sums:
        // corners = 4, edges (non-corner) = 6, interior = 9.
        Conv2D conv(1, 1);
        auto& w = conv.weights();
        std::fill(w.begin(), w.end(), 1.0f);
        conv.bias()[0] = 0.0f;

        std::vector<float> input(4 * 4, 1.0f);
        auto out = conv.forward(input, 1, 4, 4);

        auto expected_at = [](std::size_t h, std::size_t w) {
            const bool corner = (h == 0 || h == 3) && (w == 0 || w == 3);
            const bool edge = (h == 0 || h == 3 || w == 0 || w == 3);
            if (corner) return 4.0f;
            if (edge) return 6.0f;
            return 9.0f;
        };

        for (std::size_t h = 0; h < 4; ++h) {
            for (std::size_t w = 0; w < 4; ++w) {
                const std::size_t idx = h * 4 + w;
                const float exp = expected_at(h, w);
                if (!nearly_equal(out[idx], exp)) {
                    std::cerr << "All-ones conv mismatch at (" << h << "," << w
                              << "): got " << out[idx] << " expected " << exp << "\n";
                    return 1;
                }
            }
        }
        std::cout << "Conv all-ones test: OK\n";
    }

    std::cout << "All Conv2D sanity tests passed.\n";
    return 0;
}
