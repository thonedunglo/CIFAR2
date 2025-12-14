#include "autoencoder.h"

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <fstream>

namespace {
inline float he_std(std::size_t fan_in) {
    return std::sqrt(2.0f / static_cast<float>(fan_in));
}
}  // namespace

AutoencoderCPU::AutoencoderCPU()
    : conv1_(3, 256),
      conv2_(256, 128),
      conv3_(128, 128),
      conv4_(128, 256),
      conv5_(256, 3) {
    init_weights();
}

void AutoencoderCPU::init_conv_weights(Conv2D& conv, float scale,
                                       std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, scale);
    auto& w = conv.weights();
    for (auto& v : w) {
        v = dist(rng);
    }
    auto& b = conv.bias();
    std::fill(b.begin(), b.end(), 0.0f);
}

void AutoencoderCPU::init_weights(unsigned int seed) {
    std::mt19937 rng(seed);
    init_conv_weights(conv1_, he_std(3 * 3 * 3), rng);       // in=3
    init_conv_weights(conv2_, he_std(256 * 3 * 3), rng);     // in=256
    init_conv_weights(conv3_, he_std(128 * 3 * 3), rng);     // in=128
    init_conv_weights(conv4_, he_std(128 * 3 * 3), rng);     // in=128
    init_conv_weights(conv5_, he_std(256 * 3 * 3), rng);     // in=256
}

const std::vector<float>& AutoencoderCPU::encode(const std::vector<float>& input,
                                                 std::size_t batch) {
    cached_batch_ = batch;
    // Input: N x 3 x 32 x 32
    act1_ = relu1_.forward(conv1_.forward(input, batch, 32, 32));    // N x 256 x 32 x 32
    act2_ = pool1_.forward(act1_, batch, 256, 32, 32);               // N x 256 x 16 x 16
    act3_ = relu2_.forward(conv2_.forward(act2_, batch, 16, 16));    // N x 128 x 16 x 16
    act4_ = pool2_.forward(act3_, batch, 128, 16, 16);               // N x 128 x 8 x 8
    return act4_;
}

const std::vector<float>& AutoencoderCPU::forward(const std::vector<float>& input,
                                                  std::size_t batch) {
    const auto& latent = encode(input, batch);  // act4_
    act5_ = relu3_.forward(conv3_.forward(latent, batch, 8, 8));     // N x 128 x 8 x 8
    act6_ = up1_.forward(act5_, batch, 128, 8, 8);                   // N x 128 x 16 x 16
    act7_ = relu4_.forward(conv4_.forward(act6_, batch, 16, 16));    // N x 256 x 16 x 16
    act8_ = up2_.forward(act7_, batch, 256, 16, 16);                 // N x 256 x 32 x 32
    act9_ = conv5_.forward(act8_, batch, 32, 32);                    // N x 3 x 32 x 32
    return act9_;
}

std::vector<float> AutoencoderCPU::extract_features(
    const std::vector<float>& input, std::size_t batch) {
    const auto& latent = encode(input, batch);
    return latent;  // copy out
}

std::vector<float> AutoencoderCPU::backward(const std::vector<float>& grad_output) {
    if (act9_.empty()) {
        throw std::runtime_error("Backward called before forward");
    }
    const std::size_t expected = cached_batch_ * 3 * 32 * 32;
    if (grad_output.size() != expected) {
        throw std::runtime_error("grad_output size mismatch in backward");
    }

    // Clear previous grads.
    zero_grad();

    // Decoder path backward.
    auto grad8 = conv5_.backward(grad_output, cached_batch_, 32, 32);  // wrt act8_
    auto grad7 = up2_.backward(grad8);                                 // wrt act7_
    auto grad6 = relu4_.backward(grad7);                               // wrt act6_
    auto grad5 = conv4_.backward(grad6, cached_batch_, 16, 16);        // wrt act6 input (act6_)
    auto grad4 = up1_.backward(grad5);                                 // wrt act5_
    auto grad3 = relu3_.backward(grad4);                               // wrt act4_
    auto grad2 = conv3_.backward(grad3, cached_batch_, 8, 8);          // wrt act4_ input (act4_)

    // Encoder path backward.
    auto grad1 = pool2_.backward(grad2);                               // wrt act3_
    auto grad1a = relu2_.backward(grad1);                              // wrt act2_
    auto grad0 = conv2_.backward(grad1a, cached_batch_, 16, 16);       // wrt act2_ input (act2_)
    auto gradp1 = pool1_.backward(grad0);                              // wrt act1_
    auto gradp1a = relu1_.backward(gradp1);                            // wrt input to relu1 (conv1 output)
    auto grad_input = conv1_.backward(gradp1a, cached_batch_, 32, 32); // wrt original input

    return grad_input;
}

void AutoencoderCPU::zero_grad() {
    conv1_.zero_grad();
    conv2_.zero_grad();
    conv3_.zero_grad();
    conv4_.zero_grad();
    conv5_.zero_grad();
}

void AutoencoderCPU::step(float lr) {
    conv1_.step(lr);
    conv2_.step(lr);
    conv3_.step(lr);
    conv4_.step(lr);
    conv5_.step(lr);
}

namespace {
void write_vector(std::ofstream& ofs, const std::vector<float>& v) {
    uint32_t sz = static_cast<uint32_t>(v.size());
    ofs.write(reinterpret_cast<const char*>(&sz), sizeof(uint32_t));
    ofs.write(reinterpret_cast<const char*>(v.data()),
              static_cast<std::streamsize>(v.size() * sizeof(float)));
}

void read_vector(std::ifstream& ifs, std::vector<float>& v) {
    uint32_t sz = 0;
    ifs.read(reinterpret_cast<char*>(&sz), sizeof(uint32_t));
    if (sz != v.size()) {
        throw std::runtime_error("Weight size mismatch when loading weights");
    }
    ifs.read(reinterpret_cast<char*>(v.data()),
             static_cast<std::streamsize>(v.size() * sizeof(float)));
}
}  // namespace

void AutoencoderCPU::save_weights(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    const char magic[7] = {'A', 'E', 'C', 'K', 'P', 'T', '1'};
    ofs.write(magic, sizeof(magic));
    uint32_t version = 1;
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));

    write_vector(ofs, conv1_.weights());
    write_vector(ofs, conv1_.bias());
    write_vector(ofs, conv2_.weights());
    write_vector(ofs, conv2_.bias());
    write_vector(ofs, conv3_.weights());
    write_vector(ofs, conv3_.bias());
    write_vector(ofs, conv4_.weights());
    write_vector(ofs, conv4_.bias());
    write_vector(ofs, conv5_.weights());
    write_vector(ofs, conv5_.bias());
}

void AutoencoderCPU::load_weights(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    char magic[7];
    ifs.read(magic, sizeof(magic));
    const char expected[7] = {'A', 'E', 'C', 'K', 'P', 'T', '1'};
    if (std::memcmp(magic, expected, sizeof(magic)) != 0) {
        throw std::runtime_error("Invalid checkpoint magic");
    }
    uint32_t version = 0;
    ifs.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version != 1) {
        throw std::runtime_error("Unsupported checkpoint version");
    }

    read_vector(ifs, conv1_.weights());
    read_vector(ifs, conv1_.bias());
    read_vector(ifs, conv2_.weights());
    read_vector(ifs, conv2_.bias());
    read_vector(ifs, conv3_.weights());
    read_vector(ifs, conv3_.bias());
    read_vector(ifs, conv4_.weights());
    read_vector(ifs, conv4_.bias());
    read_vector(ifs, conv5_.weights());
    read_vector(ifs, conv5_.bias());
}
