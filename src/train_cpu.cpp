#include "autoencoder.h"
#include "dataset.h"
#include "layers.h"
#include "train_config.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #include <psapi.h>
    // Link thư viện Psapi cho Windows (Visual Studio)
    #pragma comment(lib, "psapi.lib") 
#else
    #include <sys/resource.h>
    #include <unistd.h>
#endif

// Hàm lấy lượng RAM tối đa đã sử dụng (Peak Memory Usage) tính bằng MB
double get_peak_memory_usage_mb() {
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        // PeakWorkingSetSize trả về bytes -> đổi sang MB
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
#else
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    // Linux: ru_maxrss trả về Kilobytes. macOS: trả về Bytes.
    #ifdef __APPLE__
        return static_cast<double>(r_usage.ru_maxrss) / (1024.0 * 1024.0);
    #else // Linux
        return static_cast<double>(r_usage.ru_maxrss) / 1024.0;
    #endif
#endif
    return 0.0;
}
int main(int argc, char** argv) {
    std::string data_dir = "cifar-10-batches-bin";
    if (argc > 1) {
        data_dir = argv[1];
    }

    CIFAR10Dataset ds(data_dir);
    ds.load();

    TrainConfig cfg;
    AutoencoderCPU ae;
    MSELoss mse;

    // Load checkpoint if requested.
    if (cfg.load_checkpoint) {
        ae.load_weights("ae_checkpoint.bin");
        std::cout << "Loaded checkpoint from ae_checkpoint.bin\n";
    }

    const std::size_t train_limit =
        (cfg.sample > 0) ? std::min(cfg.sample, ds.train_size()) : ds.train_size();
    const std::size_t num_batches =
        (train_limit + cfg.batch_size - 1) / cfg.batch_size;

    std::vector<double> epoch_times;
    double last_avg_loss = 0.0;

    for (std::size_t epoch = 0; epoch < cfg.epochs; ++epoch) {
        double loss_sum = 0.0;
        auto start = std::chrono::high_resolution_clock::now();

        for (std::size_t b = 0; b < num_batches; ++b) {
            const std::size_t start_idx = b * cfg.batch_size;
            const std::size_t end_idx = std::min(start_idx + cfg.batch_size, train_limit);
            const std::size_t batch_sz = end_idx - start_idx;

            Batch batch;
            batch.batch_size = batch_sz;
            batch.images.resize(batch_sz * 3 * 32 * 32);
            batch.labels.resize(batch_sz);
            const auto& all_images = ds.train_images();
            const auto& all_labels = ds.train_labels();
            std::copy(all_images.begin() + start_idx * 3 * 32 * 32,
                      all_images.begin() + end_idx * 3 * 32 * 32,
                      batch.images.begin());
            std::copy(all_labels.begin() + start_idx, all_labels.begin() + end_idx,
                      batch.labels.begin());

            const auto& out = ae.forward(batch.images, batch.batch_size);
            float loss = mse.forward(out, batch.images);
            auto grad_out = mse.backward(out, batch.images);

            ae.backward(grad_out);
            ae.step(cfg.lr);

            loss_sum += loss;

            if (cfg.log_interval > 0 && (b + 1) % cfg.log_interval == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_b = now - start;
                double current_mem = get_peak_memory_usage_mb();
                std::cout << "  Batch " << (b + 1) << "/" << num_batches
                          << " - loss: " << loss
                          << " - mem: " << std::setprecision(2) << current_mem << " MB"
                          << " - elapsed: " << elapsed_b.count() << "s\n";
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double avg_loss = loss_sum / static_cast<double>(num_batches);
        std::cout << "Epoch " << (epoch + 1) << "/" << cfg.epochs
                  << " - avg loss: " << avg_loss
                  << " - time: " << elapsed.count() << "s\n";
        last_avg_loss = avg_loss;
        epoch_times.push_back(elapsed.count());
    }

    // Average epoch time
    double avg_epoch_time =
        epoch_times.empty()
            ? 0.0
            : std::accumulate(epoch_times.begin(), epoch_times.end(), 0.0) /
                  static_cast<double>(epoch_times.size());

    // Save checkpoint after training
    ae.save_weights("ae_checkpoint.bin");
    std::cout << "Training finished. Weights saved to ae_checkpoint.bin\n";

    // Write log file
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&tm_now, &now_c);
#else
    localtime_r(&now_c, &tm_now);
#endif
    double peak_memory_mb = get_peak_memory_usage_mb();
    std::ofstream logf("log.txt", std::ios::out | std::ios::app);
    // std::ofstream logf("log_cpu.txt", std::ios::out);
    if (logf) {
        logf << "==============\n";
        logf << "<<<General>>>\n";
        logf << "Time: " << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S") << "\n";
        logf << "Device: CPU\n";
        logf << "Optimization: None\n";
        logf << "<<<Input>>>\n";
        logf << "Sample: " << train_limit << "\n";
        logf << "Test_sample: "
             << ((cfg.test_sample > 0) ? cfg.test_sample : ds.test_size()) << "\n";
        logf << "Batch_size: " << cfg.batch_size << "\n";
        logf << "Epochs: " << cfg.epochs << "\n";
        logf << "Log_interval: " << cfg.log_interval << "\n";
        logf << "Lr: " << cfg.lr << "\n";
        logf << "Load_checkpoint: " << (cfg.load_checkpoint ? 1 : 0) << "\n";
        logf << "<<<Result>>>\n";
        logf << "Last_epoch_loss: " << last_avg_loss << "\n";
        logf << "Avg_epoch_time: " << avg_epoch_time << "\n";
        logf << "Memory_usage: " << peak_memory_mb << " MB\n";
        logf << "==============\n";
    } else {
        std::cerr << "Warning: failed to write log.txt\n";
    }

    return 0;
}
