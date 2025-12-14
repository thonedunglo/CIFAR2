#include "autoencoder.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    // Create model A, init weights, save.
    AutoencoderCPU a;
    const std::string path = "tmp_checkpoint.bin";
    a.save_weights(path);

    // Load into model B and compare.
    AutoencoderCPU b;
    b.load_weights(path);

    auto cmp_vec = [](const std::vector<float>& v1, const std::vector<float>& v2) {
        if (v1.size() != v2.size()) return false;
        for (std::size_t i = 0; i < v1.size(); ++i) {
            if (v1[i] != v2[i]) return false;
        }
        return true;
    };

    assert(cmp_vec(a.conv1().weights(), b.conv1().weights()));
    assert(cmp_vec(a.conv1().bias(), b.conv1().bias()));
    assert(cmp_vec(a.conv2().weights(), b.conv2().weights()));
    assert(cmp_vec(a.conv2().bias(), b.conv2().bias()));
    assert(cmp_vec(a.conv3().weights(), b.conv3().weights()));
    assert(cmp_vec(a.conv3().bias(), b.conv3().bias()));
    assert(cmp_vec(a.conv4().weights(), b.conv4().weights()));
    assert(cmp_vec(a.conv4().bias(), b.conv4().bias()));
    assert(cmp_vec(a.conv5().weights(), b.conv5().weights()));
    assert(cmp_vec(a.conv5().bias(), b.conv5().bias()));

    std::cout << "Persistence round-trip OK.\n";
    return 0;
}
