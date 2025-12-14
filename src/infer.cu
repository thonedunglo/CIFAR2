#include "autoencoder.h"
#include "dataset.h"
#include "gpu_autoencoder.h"
#include "layers.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                      \
    do {                                                                     \
        cudaError_t err__ = (expr);                                          \
        if (err__ != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(err__));             \
        }                                                                    \
    } while (0)

namespace {
void write_ppm(const std::string& path, const std::vector<float>& chw,
               std::size_t H, std::size_t W) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open " + path);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (std::size_t y = 0; y < H; ++y) {
        for (std::size_t x = 0; x < W; ++x) {
            for (std::size_t c = 0; c < 3; ++c) {
                std::size_t idx = c * H * W + y * W + x;
                float v = std::clamp(chw[idx], 0.0f, 1.0f);
                uint8_t u = static_cast<uint8_t>(std::round(v * 255.0f));
                f.write(reinterpret_cast<const char*>(&u), 1);
            }
        }
    }
}

float mse_one(const std::vector<float>& a, const std::vector<float>& b) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        acc += d * d;
    }
    return acc / static_cast<float>(a.size());
}
}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = "cifar-10-batches-bin";
    std::string checkpoint = "ae_checkpoint_gpu.bin";
    std::size_t index = 0;
    std::string out_prefix = "infer";
    if (argc > 1) data_dir = argv[1];
    if (argc > 2) checkpoint = argv[2];
    if (argc > 3) index = static_cast<std::size_t>(std::stoul(argv[3]));
    if (argc > 4) out_prefix = argv[4];

    // Init GPU (device 0 by default).
    CUDA_CHECK(cudaSetDevice(0));

    CIFAR10Dataset ds(data_dir);
    ds.load();
    if (index >= ds.test_size()) {
        std::cerr << "Index out of range. Test size=" << ds.test_size() << "\n";
        return 1;
    }

    // Load checkpoint to CPU then copy to GPU.
    AutoencoderCPU cpu;
    cpu.load_weights(checkpoint);
    GPUAutoencoder gpu(/*batch_size_max=*/1);
    gpu.load_weights(cpu);

    // Prepare one sample.
    const std::size_t elems = 3 * 32 * 32;
    std::vector<float> input(elems);
    const auto& test_imgs = ds.test_images();
    std::copy(test_imgs.begin() + index * elems,
              test_imgs.begin() + (index + 1) * elems, input.begin());
    int label = ds.test_labels()[index];

    // H2D copy and forward.
    gpu.copy_input_to_device(input, 1);
    forward_naive(gpu, 1);

    // Copy reconstruction back.
    std::vector<float> recon(elems);
    CUDA_CHECK(cudaMemcpy(recon.data(), gpu.act9(), elems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Compute MSE for this sample.
    float mse = mse_one(input, recon);

    // Save PPMs.
    std::string orig_path = out_prefix + "_orig.ppm";
    std::string recon_path = out_prefix + "_recon.ppm";
    write_ppm(orig_path, input, 32, 32);
    write_ppm(recon_path, recon, 32, 32);

    std::cout << "Inference done for test index " << index << " (label=" << label
              << ")\n";
    std::cout << "MSE: " << mse << "\n";
    std::cout << "Saved: " << orig_path << " and " << recon_path << "\n";
    return 0;
}
