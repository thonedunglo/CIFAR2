#include "autoencoder.h"
#include "dataset.h"
#include "gpu_autoencoder.h"
#include "layers.h"
#include "train_config.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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

int main(int argc, char** argv) {
    std::string data_dir = "cifar-10-batches-bin";
    if (argc > 1) {
        data_dir = argv[1];
    }

    // Manually set GPU device ID if needed.
    int gpu_id = 0;  // adjust this value to target another GPU
    cudaError_t dev_err = cudaSetDevice(gpu_id);
    if (dev_err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device " << gpu_id << ": "
                  << cudaGetErrorString(dev_err) << "\n";
        return 1;
    }

    CIFAR10Dataset ds(data_dir);
    ds.load();

    TrainConfig cfg;
    // GPU default: batch_size 64
    cfg.batch_size = 64;
    AutoencoderCPU cpu_model;
    GPUAutoencoder gpu(cfg.batch_size);
    gpu.load_weights(cpu_model);

    MSELoss mse;

    const std::size_t train_limit =
        (cfg.sample > 0) ? std::min(cfg.sample, ds.train_size()) : ds.train_size();
    const std::size_t num_batches =
        (train_limit + cfg.batch_size - 1) / cfg.batch_size;

    const std::size_t elems_per_img = 3 * 32 * 32;
    std::vector<float> h_input(cfg.batch_size * elems_per_img);
    std::vector<float> h_output(cfg.batch_size * elems_per_img);

    std::vector<double> epoch_times;
    double last_avg_loss = 0.0;
    double peak_mem_mb = 0.0;
    double last_copy_ms_avg = 0.0;
    double last_forward_ms_avg = 0.0;
    double last_backward_ms_avg = 0.0;
    double last_loss_ms_avg = 0.0;

    // For now, compute loss/grad on host (simple baseline).
    for (std::size_t epoch = 0; epoch < cfg.epochs; ++epoch) {
        double loss_sum = 0.0;
        double gpu_ms_sum = 0.0;
        double copy_ms_sum = 0.0;
        double sum_forward_ms = 0.0;
        double sum_backward_ms = 0.0;
        double sum_loss_ms = 0.0;
        auto start = std::chrono::high_resolution_clock::now();

        cudaEvent_t ev_start, ev_end, ev_copy_start, ev_copy_end, ev_fwd_start,
            ev_fwd_end, ev_bwd_start, ev_bwd_end;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_end);
        cudaEventCreate(&ev_copy_start);
        cudaEventCreate(&ev_copy_end);
        cudaEventCreate(&ev_fwd_start);
        cudaEventCreate(&ev_fwd_end);
        cudaEventCreate(&ev_bwd_start);
        cudaEventCreate(&ev_bwd_end);

        for (std::size_t b = 0; b < num_batches; ++b) {
            const std::size_t start_idx = b * cfg.batch_size;
            const std::size_t end_idx = std::min(start_idx + cfg.batch_size, train_limit);
            const std::size_t batch_sz = end_idx - start_idx;

            // Track memory usage snapshot before batch.
            size_t free_mem = 0, total_mem = 0;
            double current_mem_mb = 0.0;
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
                current_mem_mb = static_cast<double>(total_mem - free_mem) /
                                 (1024.0 * 1024.0);
                peak_mem_mb = std::max(peak_mem_mb, current_mem_mb);
            }

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

            cudaEventRecord(ev_start);
            // Copy host -> device.
            std::memcpy(h_input.data(), batch.images.data(),
                        batch_sz * elems_per_img * sizeof(float));
            cudaEventRecord(ev_copy_start);
            CUDA_CHECK(cudaMemcpy(gpu.input_buf(), h_input.data(),
                                  batch_sz * elems_per_img * sizeof(float),
                                  cudaMemcpyHostToDevice));
            cudaEventRecord(ev_copy_end);
            gpu.zero_grads();

            cudaEventRecord(ev_fwd_start);
            forward_naive(gpu, batch_sz);
            cudaEventRecord(ev_fwd_end);

            auto t_loss0 = std::chrono::high_resolution_clock::now();

            // Compute Loss on GPU (returns scalar value to host)
            float loss = gpu.compute_mse_loss(batch_sz);
            loss_sum += loss;

            // Compute Loss Gradient on GPU (writes directly to device memory)
            gpu.compute_mse_gradient(batch_sz);

            auto t_loss1 = std::chrono::high_resolution_clock::now();

            cudaEventRecord(ev_bwd_start);
            // Backward pipeline: conv5 -> up2 -> relu4 -> conv4 -> up1 -> relu3
            // -> conv3 -> pool2 -> relu2 -> conv2 -> pool1 -> relu1 -> conv1.
            conv2d_backward_naive(gpu.act8(), gpu.g_act9(), gpu.w5(),
                                  gpu.g_act8(), gpu.gw5(), gpu.gb5(),
                                  batch_sz, 256, 3, 32, 32);
            upsample2x2_backward_naive(gpu.g_act8(), gpu.g_act7(), batch_sz, 256, 16, 16);
            relu_backward_naive(gpu.g_act7(), gpu.act7(), gpu.g_act7(), batch_sz * 256 * 16 * 16);
            conv2d_backward_naive(gpu.act6(), gpu.g_act7(), gpu.w4(),
                                  gpu.g_act6(), gpu.gw4(), gpu.gb4(),
                                  batch_sz, 128, 256, 16, 16);
            upsample2x2_backward_naive(gpu.g_act6(), gpu.g_act5(), batch_sz, 128, 8, 8);
            relu_backward_naive(gpu.g_act5(), gpu.act5(), gpu.g_act5(), batch_sz * 128 * 8 * 8);
            conv2d_backward_naive(gpu.act4(), gpu.g_act5(), gpu.w3(),
                                  gpu.g_act4(), gpu.gw3(), gpu.gb3(),
                                  batch_sz, 128, 128, 8, 8);
            maxpool2x2_backward_naive(gpu.g_act4(), gpu.pool2_indices(), gpu.g_act3(), batch_sz, 128, 16, 16);
            relu_backward_naive(gpu.g_act3(), gpu.act3(), gpu.g_act3(), batch_sz * 128 * 16 * 16);
            conv2d_backward_naive(gpu.act2(), gpu.g_act3(), gpu.w2(),
                                  gpu.g_act2(), gpu.gw2(), gpu.gb2(),
                                  batch_sz, 256, 128, 16, 16);
            maxpool2x2_backward_naive(gpu.g_act2(), gpu.pool1_indices(), gpu.g_act1(), batch_sz, 256, 32, 32);
            relu_backward_naive(gpu.g_act1(), gpu.act1(), gpu.g_act1(), batch_sz * 256 * 32 * 32);
            conv2d_backward_naive(gpu.input_buf(), gpu.g_act1(), gpu.w1(),
                                  gpu.g_input(), gpu.gw1(), gpu.gb1(),
                                  batch_sz, 3, 256, 32, 32);

            gpu.step(cfg.lr);
            cudaEventRecord(ev_bwd_end);

            cudaEventRecord(ev_end);
            cudaEventSynchronize(ev_end);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_end);
            float copy_ms = 0.0f;
            cudaEventElapsedTime(&copy_ms, ev_copy_start, ev_copy_end);
            float fwd_ms = 0.0f;
            cudaEventElapsedTime(&fwd_ms, ev_fwd_start, ev_fwd_end);
            float bwd_ms = 0.0f;
            cudaEventElapsedTime(&bwd_ms, ev_bwd_start, ev_bwd_end);
            gpu_ms_sum += ms;
            copy_ms_sum += copy_ms;
            sum_forward_ms += fwd_ms;
            sum_backward_ms += bwd_ms;
            std::chrono::duration<double, std::milli> loss_ms = t_loss1 - t_loss0;
            sum_loss_ms += loss_ms.count();

            if (cfg.log_interval > 0 && (b + 1) % cfg.log_interval == 0) {
                std::cout << "  Batch " << (b + 1) << "/" << num_batches
                          << " - loss: " << loss
                          << " - gpu_ms: " << ms
                          << " - copy_ms: " << copy_ms
                          << " - fwd_ms: " << fwd_ms
                          << " - bwd_ms: " << bwd_ms
                          << " - loss_ms: " << loss_ms.count()
                          << " - mem_used_mb: " << current_mem_mb << "\n";
            }
        }

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);
        cudaEventDestroy(ev_copy_start);
        cudaEventDestroy(ev_copy_end);
        cudaEventDestroy(ev_fwd_start);
        cudaEventDestroy(ev_fwd_end);
        cudaEventDestroy(ev_bwd_start);
        cudaEventDestroy(ev_bwd_end);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double avg_loss = loss_sum / static_cast<double>(num_batches);
        double avg_gpu_ms = gpu_ms_sum / static_cast<double>(num_batches);
        std::cout << "GPU Epoch " << (epoch + 1) << "/" << cfg.epochs
                  << " - avg loss: " << avg_loss
                  << " - time: " << elapsed.count() << "s"
                  << " - avg gpu time per batch: " << avg_gpu_ms
                  << " ms"
                  << " - avg copy time per batch: "
                  << (copy_ms_sum / static_cast<double>(num_batches))
                  << " ms"
                  << " - avg forward ms: "
                  << (sum_forward_ms / static_cast<double>(num_batches))
                  << " - avg backward ms: "
                  << (sum_backward_ms / static_cast<double>(num_batches))
                  << " - avg loss ms: "
                  << (sum_loss_ms / static_cast<double>(num_batches))
                  << " ms\n";
        epoch_times.push_back(elapsed.count());
        last_avg_loss = avg_loss;
        last_copy_ms_avg = copy_ms_sum / static_cast<double>(num_batches);
        last_forward_ms_avg = sum_forward_ms / static_cast<double>(num_batches);
        last_backward_ms_avg = sum_backward_ms / static_cast<double>(num_batches);
        last_loss_ms_avg = sum_loss_ms / static_cast<double>(num_batches);
    }

    // host buffers are std::vector; no explicit free needed
    gpu.save_weights(cpu_model);
    cpu_model.save_weights("ae_checkpoint_gpu.bin");
    std::cout << "GPU training finished. Weights saved to ae_checkpoint_gpu.bin\n";

    // Write log file similar format to CPU.
    double avg_epoch_time =
        epoch_times.empty()
            ? 0.0
            : std::accumulate(epoch_times.begin(), epoch_times.end(), 0.0) /
                  static_cast<double>(epoch_times.size());

    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&tm_now, &now_c);
#else
    localtime_r(&now_c, &tm_now);
#endif
    std::ofstream logf("log.txt", std::ios::out | std::ios::app);
    if (logf) {
        logf << "==============\n";
        logf << "<<<General>>>\n";
        logf << "Time: " << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S") << "\n";
        logf << "Device: GPU\n";
        logf << "Optimization: Version 2\n";
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
        logf << "Memory_usage: " << peak_mem_mb << " MB\n";
        logf << "H2D_copy: "
             << last_copy_ms_avg << " ms\n";
        logf << "Forward: "
             << last_forward_ms_avg << " ms\n";
        logf << "Backward: "
             << last_backward_ms_avg << " ms\n";
        logf << "Loss_ms: "
             << last_loss_ms_avg << " ms\n";
        logf << "==============\n";
    } else {
        std::cerr << "Warning: failed to write log.txt\n";
    }
    return 0;
}
