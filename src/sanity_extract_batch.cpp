#include "autoencoder.h"
#include "dataset.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string data_dir = "cifar-10-batches-bin";
    std::string checkpoint = "ae_checkpoint.bin";
    std::size_t batch_size = 4;
    bool train_split = true;

    if (argc > 1) data_dir = argv[1];
    if (argc > 2) checkpoint = argv[2];
    if (argc > 3) batch_size = static_cast<std::size_t>(std::stoul(argv[3]));
    if (argc > 4) {
        std::string split = argv[4];
        train_split = (split != "test");
    }

    CIFAR10Dataset ds(data_dir);
    ds.load();

    if (batch_size == 0 || batch_size > (train_split ? ds.train_size() : ds.test_size())) {
        std::cerr << "Invalid batch size\n";
        return 1;
    }

    AutoencoderCPU ae;
    ae.load_weights(checkpoint);

    auto batch = ds.get_batch(train_split, batch_size, /*shuffle=*/false);
    auto features = ae.extract_features(batch.images, batch.batch_size);

    std::size_t expected = batch.batch_size * 128 * 8 * 8;
    assert(features.size() == expected);

    std::cout << "Loaded checkpoint: " << checkpoint << "\n";
    std::cout << "Split: " << (train_split ? "train" : "test")
              << ", batch_size=" << batch.batch_size << "\n";
    std::cout << "Features size: " << features.size() << " ("
              << expected << " expected)\n";
    std::cout << "First label: " << batch.labels[0] << "\n";
    std::cout << "First 5 feature values of sample 0: ";
    for (int i = 0; i < 5 && i < static_cast<int>(features.size()); ++i) {
        std::cout << features[i] << (i == 4 ? "" : ", ");
    }
    std::cout << "\n";
    std::cout << "Batch feature extraction sanity passed.\n";
    return 0;
}
