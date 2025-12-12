#include "autoencoder.h"
#include "dataset.h"
#include "gpu_autoencoder.h"
#include "layers.h"
#include "train_config.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cuda.h>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string data_dir = "cifar-10-batches-bin";
    if (argc > 1) {
        data_dir = argv[1];
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

    std::vector<float> h_output(cfg.batch_size * 3 * 32 * 32);
    std::vector<float> h_grad(cfg.batch_size * 3 * 32 * 32);

    // For now, compute loss/grad on host (simple baseline).
    for (std::size_t epoch = 0; epoch < cfg.epochs; ++epoch) {
        double loss_sum = 0.0;
        double gpu_ms_sum = 0.0;
        auto start = std::chrono::high_resolution_clock::now();

        cudaEvent_t ev_start, ev_end;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_end);

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

            cudaEventRecord(ev_start);
            gpu.copy_input_to_device(batch.images, batch_sz);
            gpu.zero_grads();

            forward_naive(gpu, batch_sz);

            const std::size_t elems = batch_sz * 3 * 32 * 32;
            cudaMemcpy(h_output.data(), gpu.act9(), elems * sizeof(float),
                       cudaMemcpyDeviceToHost);

            float loss = mse.forward(h_output, batch.images);
            auto grad_host = mse.backward(h_output, batch.images);
            loss_sum += loss;

            cudaMemcpy(gpu.g_act9(), grad_host.data(),
                       elems * sizeof(float), cudaMemcpyHostToDevice);

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

            cudaEventRecord(ev_end);
            cudaEventSynchronize(ev_end);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_end);
            gpu_ms_sum += ms;

            if (cfg.log_interval > 0 && (b + 1) % cfg.log_interval == 0) {
                std::cout << "  Batch " << (b + 1) << "/" << num_batches
                          << " - loss: " << loss
                          << " - gpu_ms: " << ms << "\n";
            }
        }

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double avg_loss = loss_sum / static_cast<double>(num_batches);
        double avg_gpu_ms = gpu_ms_sum / static_cast<double>(num_batches);
        std::cout << "GPU Epoch " << (epoch + 1) << "/" << cfg.epochs
                  << " - avg loss: " << avg_loss
                  << " - time: " << elapsed.count() << "s"
                  << " - avg gpu time per batch: " << avg_gpu_ms << " ms\n";
    }

    gpu.save_weights(cpu_model);
    cpu_model.save_weights("ae_checkpoint_gpu.bin");
    std::cout << "GPU training finished. Weights saved to ae_checkpoint_gpu.bin\n";
    return 0;
}
