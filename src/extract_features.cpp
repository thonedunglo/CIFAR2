#include "autoencoder.h"
#include "dataset.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Args {
    std::string data_dir = "cifar-10-batches-bin";
    std::string checkpoint = "ae_checkpoint.bin";
    std::string out_prefix = "features";
    std::size_t batch_size = 128;
    std::size_t sample = 0;       // 0 = full train
    std::size_t log_interval = 50;
    std::size_t test_sample = 0;  // 0 = full test
};

bool parse_args(int argc, char** argv, Args& args) {
    if (argc > 1) args.data_dir = argv[1];
    if (argc > 2) args.checkpoint = argv[2];
    if (argc > 3) args.out_prefix = argv[3];
    if (argc > 4) args.batch_size = static_cast<std::size_t>(std::stoul(argv[4]));
    if (argc > 5) args.sample = static_cast<std::size_t>(std::stoul(argv[5]));
    if (argc > 6) args.log_interval = static_cast<std::size_t>(std::stoul(argv[6]));
    if (argc > 7) args.test_sample = static_cast<std::size_t>(std::stoul(argv[7]));
    return true;
}

void write_all(std::ofstream& ofs, const std::vector<float>& buf, std::size_t elems) {
    ofs.write(reinterpret_cast<const char*>(buf.data()),
              static_cast<std::streamsize>(elems * sizeof(float)));
}

void write_labels(std::ofstream& ofs, const std::vector<int>& labels, std::size_t count) {
    // write as uint8 since labels are 0-9
    for (std::size_t i = 0; i < count; ++i) {
        uint8_t v = static_cast<uint8_t>(labels[i]);
        ofs.write(reinterpret_cast<const char*>(&v), sizeof(uint8_t));
    }
}

void process_split(const std::vector<float>& images, const std::vector<int>& labels,
                   AutoencoderCPU& ae, std::size_t total, std::size_t batch_size,
                   const std::string& feature_path, const std::string& label_path,
                   std::size_t log_interval, double& time_sum_out,
                   std::size_t& batches_out) {
    std::ofstream feat_out(feature_path, std::ios::binary | std::ios::trunc);
    std::ofstream lbl_out(label_path, std::ios::binary | std::ios::trunc);
    if (!feat_out || !lbl_out) {
        throw std::runtime_error("Failed to open output files");
    }

    const std::size_t elems_per_image = 3 * 32 * 32;
    const std::size_t latent_dim = 128 * 8 * 8;  // 8192
    std::vector<float> batch_imgs;
    std::vector<int> batch_labels;
    std::vector<float> latent;

    const std::size_t num_batches = (total + batch_size - 1) / batch_size;
    double time_sum = 0.0;
    for (std::size_t b = 0; b < num_batches; ++b) {
        auto t0 = std::chrono::high_resolution_clock::now();

        const std::size_t start = b * batch_size;
        const std::size_t end = std::min(start + batch_size, total);
        const std::size_t bs = end - start;
        batch_imgs.resize(bs * elems_per_image);
        batch_labels.resize(bs);
        std::copy(images.begin() + start * elems_per_image,
                  images.begin() + end * elems_per_image,
                  batch_imgs.begin());
        std::copy(labels.begin() + start, labels.begin() + end, batch_labels.begin());

        latent = ae.extract_features(batch_imgs, bs);  // returns copy
        if (latent.size() != bs * latent_dim) {
            throw std::runtime_error("Latent size mismatch");
        }
        write_all(feat_out, latent, latent.size());
        write_labels(lbl_out, batch_labels, bs);

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        time_sum += dt.count();

        if ((log_interval > 0 && (b + 1) % log_interval == 0) || b == num_batches - 1) {
            std::cout << "  Batch " << (b + 1) << "/" << num_batches
                      << " processed - time: " << dt.count() << " s\n";
        }
    }
    double avg = time_sum / static_cast<double>(num_batches);
    std::cout << "  Avg time per batch: " << avg << " s\n";
    time_sum_out += time_sum;
    batches_out += num_batches;
}

int main(int argc, char** argv) {
    Args args;
    parse_args(argc, argv, args);

    CIFAR10Dataset ds(args.data_dir);
    ds.load();

    AutoencoderCPU ae;
    ae.load_weights(args.checkpoint);

    const std::size_t train_limit =
        (args.sample > 0) ? std::min(args.sample, ds.train_size()) : ds.train_size();
    const std::size_t test_limit =
        (args.test_sample > 0) ? std::min(args.test_sample, ds.test_size()) : ds.test_size();

    auto start = std::chrono::high_resolution_clock::now();
    double time_sum = 0.0;
    std::size_t total_batches = 0;
    std::cout << "Extracting train features...\n";
    process_split(ds.train_images(), ds.train_labels(), ae, train_limit,
                  args.batch_size, args.out_prefix + "_train_features.bin",
                  args.out_prefix + "_train_labels.bin", args.log_interval,
                  time_sum, total_batches);

    std::cout << "Extracting test features...\n";
    process_split(ds.test_images(), ds.test_labels(), ae, test_limit,
                  args.batch_size, args.out_prefix + "_test_features.bin",
                  args.out_prefix + "_test_labels.bin", args.log_interval,
                  time_sum, total_batches);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double avg_time = (total_batches > 0) ? (time_sum / static_cast<double>(total_batches)) : 0.0;
    std::cout << "Done. Total time: " << elapsed.count() << " s\n";
    std::cout << "Outputs:\n  " << args.out_prefix << "_train_features.bin\n  "
              << args.out_prefix << "_train_labels.bin\n  " << args.out_prefix
              << "_test_features.bin\n  " << args.out_prefix << "_test_labels.bin\n";

    // Append log
    std::ofstream logf("extract_log.txt", std::ios::app);
    if (logf) {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        logf << "================\n";
        logf << "Datetime: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "\n";
        logf << "Device: CPU\n";
        logf << "Train sample: " << train_limit << "\n";
        logf << "Test sample: " << test_limit << "\n";
        logf << "Batch size: " << args.batch_size << "\n";
        logf << "Total time: " << elapsed.count() << " s\n";
        logf << "Avg time per batch: " << avg_time << " s\n";
        logf << "Outputs:\n";
        logf << "  " << args.out_prefix << "_train_features.bin\n";
        logf << "  " << args.out_prefix << "_train_labels.bin\n";
        logf << "  " << args.out_prefix << "_test_features.bin\n";
        logf << "  " << args.out_prefix << "_test_labels.bin\n";
        logf << "================\n";
    }
    return 0;
}
