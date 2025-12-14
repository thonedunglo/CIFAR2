#include "autoencoder.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

int main(int argc, char** argv) {
    std::string path = "ae_checkpoint.bin";
    if (argc > 1) {
        path = argv[1];
    }

    AutoencoderCPU ae;
    try {
        ae.load_weights(path);
    } catch (const std::exception& ex) {
        std::cerr << "Failed to load checkpoint: " << ex.what() << "\n";
        return 1;
    }

    auto check_shape = [](const std::vector<float>& w, std::size_t expect) {
        if (w.size() != expect) {
            std::cerr << "Size mismatch: got " << w.size()
                      << " expected " << expect << "\n";
            return false;
        }
        return true;
    };

    bool ok = true;
    ok &= check_shape(ae.conv1().weights(), 256 * 3 * 3 * 3);
    ok &= check_shape(ae.conv2().weights(), 128 * 256 * 3 * 3);
    ok &= check_shape(ae.conv3().weights(), 128 * 128 * 3 * 3);
    ok &= check_shape(ae.conv4().weights(), 256 * 128 * 3 * 3);
    ok &= check_shape(ae.conv5().weights(), 3 * 256 * 3 * 3);
    ok &= check_shape(ae.conv1().bias(), 256);
    ok &= check_shape(ae.conv2().bias(), 128);
    ok &= check_shape(ae.conv3().bias(), 128);
    ok &= check_shape(ae.conv4().bias(), 256);
    ok &= check_shape(ae.conv5().bias(), 3);

    auto stats = [](const std::vector<float>& v) {
        double sum = 0.0;
        double minv = v.empty() ? 0.0 : v[0];
        double maxv = v.empty() ? 0.0 : v[0];
        for (float x : v) {
            sum += x;
            if (x < minv) minv = x;
            if (x > maxv) maxv = x;
        }
        double mean = v.empty() ? 0.0 : sum / v.size();
        return std::make_tuple(mean, minv, maxv);
    };

    auto [m1, min1, max1] = stats(ae.conv1().weights());
    auto [m5, min5, max5] = stats(ae.conv5().weights());

    std::cout << "Loaded checkpoint: " << path << "\n";
    if (!ok) {
        std::cerr << "Shape check failed.\n";
        return 1;
    }
    std::cout << "Conv1 weights: mean=" << m1 << " min=" << min1
              << " max=" << max1 << "\n";
    std::cout << "Conv5 weights: mean=" << m5 << " min=" << min5
              << " max=" << max5 << "\n";

    // Simple finite check.
    auto finite_check = [](const std::vector<float>& v) {
        for (float x : v) {
            if (!std::isfinite(x)) return false;
        }
        return true;
    };
    if (!finite_check(ae.conv1().weights()) || !finite_check(ae.conv5().weights())) {
        std::cerr << "Non-finite values detected.\n";
        return 1;
    }

    std::cout << "Checkpoint sanity OK.\n";
    return 0;
}
