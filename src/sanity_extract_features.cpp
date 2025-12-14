#include "autoencoder.h"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    AutoencoderCPU ae;
    const std::size_t batch = 3;
    std::vector<float> input(batch * 3 * 32 * 32, 0.0f);

    auto feats = ae.extract_features(input, batch);
    std::size_t expected = batch * 128 * 8 * 8;
    assert(feats.size() == expected);

    std::cout << "Extracted features size: " << feats.size() << " (expected "
              << expected << ")\n";
    std::cout << "Extract features sanity passed.\n";
    return 0;
}
