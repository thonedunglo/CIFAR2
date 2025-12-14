#include "autoencoder.h"

#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) <= eps;
}

std::pair<float, float> mean_std(const std::vector<float>& v) {
    if (v.empty()) return {0.0f, 0.0f};
    double sum = 0.0;
    for (float x : v) sum += x;
    double mean = sum / v.size();
    double var = 0.0;
    for (float x : v) {
        double d = x - mean;
        var += d * d;
    }
    var /= v.size();
    return {static_cast<float>(mean), static_cast<float>(std::sqrt(var))};
}

int main() {
    AutoencoderCPU ae;

    auto check_shape = [](const Conv2D& conv, std::size_t out_c,
                          std::size_t in_c) {
        const auto& w = conv.weights();
        const auto& b = conv.bias();
        assert(w.size() == out_c * in_c * 3 * 3);
        assert(b.size() == out_c);
    };

    check_shape(ae.conv1(), 256, 3);
    check_shape(ae.conv2(), 128, 256);
    check_shape(ae.conv3(), 128, 128);
    check_shape(ae.conv4(), 256, 128);
    check_shape(ae.conv5(), 3, 256);

    // Bias should be zeros.
    for (const auto* bptr :
         {&ae.conv1().bias(), &ae.conv2().bias(), &ae.conv3().bias(),
          &ae.conv4().bias(), &ae.conv5().bias()}) {
        for (float v : *bptr) {
            assert(nearly_equal(v, 0.0f));
        }
    }

    // Weight stats: mean ~ 0, std near He init scale.
    auto [m1, s1] = mean_std(ae.conv1().weights());
    std::cout << "Conv1 weight mean=" << m1 << " std=" << s1 << "\n";

    std::cout << "Autoencoder init sanity passed.\n";
    return 0;
}
