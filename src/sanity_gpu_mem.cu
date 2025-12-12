#include "autoencoder.h"
#include "gpu_autoencoder.h"

#include <cassert>
#include <iostream>

int main() {
    const std::size_t batch_max = 64;
    AutoencoderCPU cpu_a;
    GPUAutoencoder gpu(batch_max);

    gpu.load_weights(cpu_a);

    AutoencoderCPU cpu_b;
    gpu.save_weights(cpu_b);

    auto cmp = [](const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) return false;
        }
        return true;
    };

    assert(cmp(cpu_a.conv1().weights(), cpu_b.conv1().weights()));
    assert(cmp(cpu_a.conv2().weights(), cpu_b.conv2().weights()));
    assert(cmp(cpu_a.conv3().weights(), cpu_b.conv3().weights()));
    assert(cmp(cpu_a.conv4().weights(), cpu_b.conv4().weights()));
    assert(cmp(cpu_a.conv5().weights(), cpu_b.conv5().weights()));
    assert(cmp(cpu_a.conv1().bias(), cpu_b.conv1().bias()));
    assert(cmp(cpu_a.conv2().bias(), cpu_b.conv2().bias()));
    assert(cmp(cpu_a.conv3().bias(), cpu_b.conv3().bias()));
    assert(cmp(cpu_a.conv4().bias(), cpu_b.conv4().bias()));
    assert(cmp(cpu_a.conv5().bias(), cpu_b.conv5().bias()));

    std::cout << "GPU memory copy round-trip OK\n";
    return 0;
}
