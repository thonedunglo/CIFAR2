#pragma once

#include <cstddef>

struct TrainConfig {
    std::size_t batch_size = 64;
    std::size_t epochs = 2;
    float lr = 0.002f;
    // If >0, only use the first `sample` train examples.
    std::size_t sample = 1024;
    // If >0, only use the first `test_sample` test examples (for eval later).
    std::size_t test_sample = 100;
    // Log to stdout every `log_interval` batches (0 = disable).
    std::size_t log_interval = 4;  // GPU loop will override if needed.
    // If true, load checkpoint before training.
    bool load_checkpoint = false;
};
