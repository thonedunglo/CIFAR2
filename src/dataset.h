#pragma once

#include <cstddef>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

struct Batch {
    std::vector<float> images;  // channel-first contiguous: [N, 3, 32, 32]
    std::vector<int> labels;
    std::size_t batch_size = 0;
    std::size_t channels = 3;
    std::size_t height = 32;
    std::size_t width = 32;
};

class CIFAR10Dataset {
public:
    explicit CIFAR10Dataset(std::filesystem::path root_dir, bool normalize = true);

    // Load all training and test batches + label names.
    void load();

    std::size_t train_size() const;
    std::size_t test_size() const;

    const std::vector<float>& train_images() const;
    const std::vector<int>& train_labels() const;
    const std::vector<float>& test_images() const;
    const std::vector<int>& test_labels() const;
    const std::vector<std::string>& label_names() const;

    // Return a batch; for train split, shuffles epoch boundaries by default.
    Batch get_batch(bool train_split, std::size_t batch_size, bool shuffle = true);

private:
    void load_split(const std::vector<std::string>& files,
                    std::vector<float>& images_out,
                    std::vector<int>& labels_out);
    std::vector<uint8_t> read_file_bytes(const std::filesystem::path& path) const;
    void decode_records(const std::vector<uint8_t>& buffer,
                        std::size_t num_records,
                        std::vector<float>& images_out,
                        std::vector<int>& labels_out);
    void load_label_names();

    std::filesystem::path root_dir_;
    bool normalize_;

    std::vector<float> train_images_;
    std::vector<int> train_labels_;
    std::vector<float> test_images_;
    std::vector<int> test_labels_;
    std::vector<std::string> label_names_;

    // Batching state.
    std::mt19937 rng_;
    std::vector<std::size_t> train_indices_;
    std::size_t train_cursor_ = 0;
};
