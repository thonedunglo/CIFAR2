#include "dataset.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

int main(int argc, char** argv) {
    std::string root = "cifar-10-batches-bin";
    if (argc > 1) {
        root = argv[1];
    }

    try {
        CIFAR10Dataset ds(root, /*normalize=*/true);
        ds.load();

        std::cout << "Loaded CIFAR-10 from: " << root << "\n";
        std::cout << "Train size: " << ds.train_size()
                  << ", Test size: " << ds.test_size() << "\n";
        std::cout << "Label names (10):";
        for (const auto& name : ds.label_names()) {
            std::cout << " " << name;
        }
        std::cout << "\n";

        // Check min/max over train images.
        const auto& images = ds.train_images();
        auto [min_it, max_it] =
            std::minmax_element(images.begin(), images.end());
        std::cout << "Train pixel range after normalize: [" << *min_it << ", "
                  << *max_it << "]\n";

        // Check a small batch.
        const std::size_t batch_size = 4;
        auto batch = ds.get_batch(/*train_split=*/true, batch_size, /*shuffle=*/false);
        std::cout << "Sample batch size: " << batch.batch_size << "\n";
        std::cout << "First labels:";
        for (std::size_t i = 0; i < batch.batch_size; ++i) {
            std::cout << " " << batch.labels[i];
        }
        std::cout << "\n";

        // Sanity: ensure label IDs are in [0,9].
        bool labels_ok = std::all_of(batch.labels.begin(), batch.labels.end(),
                                     [](int v) { return v >= 0 && v <= 9; });
        std::cout << "Labels in [0,9]: " << (labels_ok ? "yes" : "no") << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
