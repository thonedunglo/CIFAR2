#include "dataset.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace {
constexpr std::size_t kImageWidth = 32;
constexpr std::size_t kImageHeight = 32;
constexpr std::size_t kChannels = 3;
constexpr std::size_t kPixelsPerImage = kImageWidth * kImageHeight;
constexpr std::size_t kBytesPerImage = kChannels * kPixelsPerImage;  // 3072
constexpr std::size_t kBytesPerRecord = 1 + kBytesPerImage;          // label + image
constexpr std::size_t kRecordsPerFile = 10000;
}  // namespace

CIFAR10Dataset::CIFAR10Dataset(std::filesystem::path root_dir, bool normalize)
    : root_dir_(std::move(root_dir)),
      normalize_(normalize),
      rng_(std::random_device{}()) {}

void CIFAR10Dataset::load() {
    const std::vector<std::string> train_files = {
        "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
        "data_batch_4.bin", "data_batch_5.bin"};
    const std::vector<std::string> test_files = {"test_batch.bin"};

    load_split(train_files, train_images_, train_labels_);
    load_split(test_files, test_images_, test_labels_);
    load_label_names();

    // Prepare train indices for batching.
    train_indices_.resize(train_labels_.size());
    std::iota(train_indices_.begin(), train_indices_.end(), 0);
}

std::size_t CIFAR10Dataset::train_size() const { return train_labels_.size(); }

std::size_t CIFAR10Dataset::test_size() const { return test_labels_.size(); }

const std::vector<float>& CIFAR10Dataset::train_images() const {
    return train_images_;
}

const std::vector<int>& CIFAR10Dataset::train_labels() const {
    return train_labels_;
}

const std::vector<float>& CIFAR10Dataset::test_images() const {
    return test_images_;
}

const std::vector<int>& CIFAR10Dataset::test_labels() const {
    return test_labels_;
}

const std::vector<std::string>& CIFAR10Dataset::label_names() const {
    return label_names_;
}

Batch CIFAR10Dataset::get_batch(bool train_split, std::size_t batch_size,
                                bool shuffle) {
    const auto& images = train_split ? train_images_ : test_images_;
    const auto& labels = train_split ? train_labels_ : test_labels_;

    if (images.empty() || labels.empty()) {
        throw std::runtime_error("Dataset not loaded");
    }

    Batch batch;
    const std::size_t total = labels.size();
    const std::size_t actual_batch =
        std::min(batch_size, total - (train_split ? train_cursor_ : 0));
    batch.batch_size = actual_batch;
    batch.images.resize(actual_batch * kChannels * kPixelsPerImage);
    batch.labels.resize(actual_batch);

    if (train_split) {
        if (train_cursor_ == 0 && shuffle) {
            std::shuffle(train_indices_.begin(), train_indices_.end(), rng_);
        }

        for (std::size_t i = 0; i < actual_batch; ++i) {
            const std::size_t idx = train_indices_[train_cursor_ + i];
            const float* src = images.data() + idx * kBytesPerImage;
            float* dst = batch.images.data() + i * kBytesPerImage;
            std::copy(src, src + kBytesPerImage, dst);
            batch.labels[i] = labels[idx];
        }

        train_cursor_ += actual_batch;
        if (train_cursor_ >= total) {
            train_cursor_ = 0;  // wrap to next epoch
        }
    } else {
        // For test split, return the first actual_batch samples (no internal
        // cursor). Shuffle here only if explicitly requested.
        std::vector<std::size_t> indices(total);
        std::iota(indices.begin(), indices.end(), 0);
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), rng_);
        }
        for (std::size_t i = 0; i < actual_batch; ++i) {
            const std::size_t idx = indices[i];
            const float* src = images.data() + idx * kBytesPerImage;
            float* dst = batch.images.data() + i * kBytesPerImage;
            std::copy(src, src + kBytesPerImage, dst);
            batch.labels[i] = labels[idx];
        }
    }

    return batch;
}

void CIFAR10Dataset::load_split(const std::vector<std::string>& files,
                                std::vector<float>& images_out,
                                std::vector<int>& labels_out) {
    images_out.clear();
    labels_out.clear();
    images_out.reserve(files.size() * kRecordsPerFile * kBytesPerImage);
    labels_out.reserve(files.size() * kRecordsPerFile);

    for (const auto& file : files) {
        const auto path = root_dir_ / file;
        const auto buffer = read_file_bytes(path);
        const std::size_t num_records = buffer.size() / kBytesPerRecord;
        decode_records(buffer, num_records, images_out, labels_out);
    }
}

std::vector<uint8_t> CIFAR10Dataset::read_file_bytes(
    const std::filesystem::path& path) const {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }

    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size % kBytesPerRecord != 0 || size != static_cast<std::streamsize>(
                                             kRecordsPerFile * kBytesPerRecord)) {
        throw std::runtime_error("Unexpected file size for: " + path.string());
    }

    std::vector<uint8_t> buffer(static_cast<std::size_t>(size));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read file: " + path.string());
    }
    return buffer;
}

void CIFAR10Dataset::decode_records(const std::vector<uint8_t>& buffer,
                                    std::size_t num_records,
                                    std::vector<float>& images_out,
                                    std::vector<int>& labels_out) {
    const std::size_t current_images = images_out.size() / kBytesPerImage;
    images_out.resize((current_images + num_records) * kBytesPerImage);
    labels_out.resize(current_images + num_records);

    for (std::size_t i = 0; i < num_records; ++i) {
        const std::size_t src_offset = i * kBytesPerRecord;
        const std::size_t dst_image = (current_images + i) * kBytesPerImage;

        labels_out[current_images + i] = buffer[src_offset];

        const float norm = normalize_ ? 1.0f / 255.0f : 1.0f;
        const uint8_t* src = buffer.data() + src_offset + 1;
        float* dst = images_out.data() + dst_image;

        for (std::size_t c = 0; c < kChannels; ++c) {
            const std::size_t channel_offset = c * kPixelsPerImage;
            for (std::size_t p = 0; p < kPixelsPerImage; ++p) {
                dst[channel_offset + p] =
                    static_cast<float>(src[channel_offset + p]) * norm;
            }
        }
    }
}

void CIFAR10Dataset::load_label_names() {
    const auto path = root_dir_ / "batches.meta.txt";
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open label names: " + path.string());
    }
    label_names_.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            label_names_.push_back(line);
        }
    }
    if (label_names_.size() != 10) {
        throw std::runtime_error("Expected 10 label names, got " +
                                 std::to_string(label_names_.size()));
    }
}
