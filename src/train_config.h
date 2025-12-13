#pragma once

#include <cstddef>

struct TrainConfig {
    std::size_t batch_size = 128;
    std::size_t epochs = 10;
    float lr = 0.002f;
    // If >0, only use the first `sample` train examples.
    std::size_t sample = 50000;
    // If >0, only use the first `test_sample` test examples (for eval later).
    std::size_t test_sample = 10000;
    // Log to stdout every `log_interval` batches (0 = disable).
    std::size_t log_interval = 50;  // GPU loop will override if needed.
    // If true, load checkpoint before training.
    bool load_checkpoint = false;
};
