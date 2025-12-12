#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../third_party/libsvm/svm.h"

struct Args {
    std::string prefix = "features";
    std::size_t dim = 8192;
    std::size_t sample_train = 0;  // 0 = full
    std::size_t sample_test = 0;   // 0 = full
    double C = 10.0;               // soft-margin cost
    double gamma = 0.0;            // RBF gamma; 0 => 1/dim
    double cache_mb = 200.0;       // libsvm kernel cache
    double eps = 1e-3;             // stopping tolerance
};

bool parse_args(int argc, char** argv, Args& args) {
    if (argc > 1) args.prefix = argv[1];
    if (argc > 2) args.dim = static_cast<std::size_t>(std::stoul(argv[2]));
    if (argc > 3) args.sample_train = static_cast<std::size_t>(std::stoul(argv[3]));
    if (argc > 4) args.sample_test = static_cast<std::size_t>(std::stoul(argv[4]));
    if (argc > 5) args.C = std::stod(argv[5]);
    if (argc > 6) args.gamma = std::stod(argv[6]);
    if (argc > 7) args.cache_mb = std::stod(argv[7]);
    if (argc > 8) args.eps = std::stod(argv[8]);
    return true;
}

struct DatasetMem {
    std::vector<float> X;  // row-major
    std::vector<int> y;
    std::size_t n = 0;
    std::size_t dim = 0;
};

DatasetMem load_split(const std::string& feat_path, const std::string& lbl_path,
                      std::size_t dim, std::size_t limit = 0) {
    DatasetMem ds;
    ds.dim = dim;

    std::ifstream ffeat(feat_path, std::ios::binary);
    std::ifstream flbl(lbl_path, std::ios::binary);
    if (!ffeat || !flbl) {
        throw std::runtime_error("Failed to open feature/label file");
    }

    // Get file sizes
    ffeat.seekg(0, std::ios::end);
    std::size_t feat_bytes = static_cast<std::size_t>(ffeat.tellg());
    ffeat.seekg(0, std::ios::beg);

    flbl.seekg(0, std::ios::end);
    std::size_t lbl_bytes = static_cast<std::size_t>(flbl.tellg());
    flbl.seekg(0, std::ios::beg);

    std::size_t total_samples = feat_bytes / (dim * sizeof(float));
    if (lbl_bytes != total_samples * sizeof(uint8_t)) {
        throw std::runtime_error("Label/feature count mismatch");
    }
    if (limit > 0 && limit < total_samples) {
        total_samples = limit;
    }

    ds.n = total_samples;
    ds.X.resize(ds.n * dim);
    ds.y.resize(ds.n);

    ffeat.read(reinterpret_cast<char*>(ds.X.data()),
               static_cast<std::streamsize>(ds.n * dim * sizeof(float)));

    std::vector<uint8_t> y8(ds.n);
    flbl.read(reinterpret_cast<char*>(y8.data()),
              static_cast<std::streamsize>(ds.n * sizeof(uint8_t)));
    for (std::size_t i = 0; i < ds.n; ++i) {
        ds.y[i] = static_cast<int>(y8[i]);
    }
    return ds;
}

int main(int argc, char** argv) {
    Args args;
    parse_args(argc, argv, args);

    const std::string train_feat = args.prefix + "_train_features.bin";
    const std::string train_lbl = args.prefix + "_train_labels.bin";
    const std::string test_feat = args.prefix + "_test_features.bin";
    const std::string test_lbl = args.prefix + "_test_labels.bin";

    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "Loading data...\n";
    DatasetMem train = load_split(train_feat, train_lbl, args.dim, args.sample_train);
    DatasetMem test = load_split(test_feat, test_lbl, args.dim, args.sample_test);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt_load = t1 - t0;
    std::cout << "Loaded train: " << train.n << " samples, test: " << test.n
              << " samples. Load time: " << dt_load.count() << " s\n";

    // LibSVM expects gamma > 0.
    if (args.gamma <= 0.0) {
        args.gamma = 1.0 / static_cast<double>(args.dim);
    }

    const int num_classes = 10;

    // Build svm_problem for train split.
    svm_problem prob;
    prob.l = static_cast<int>(train.n);
    std::vector<double> y_train(train.n);
    std::vector<svm_node> nodes_train(train.n * (args.dim + 1));
    std::vector<svm_node*> x_ptrs(train.n);
    for (std::size_t i = 0; i < train.n; ++i) {
        y_train[i] = static_cast<double>(train.y[i]);
        x_ptrs[i] = &nodes_train[i * (args.dim + 1)];
        svm_node* row = x_ptrs[i];
        for (std::size_t d = 0; d < args.dim; ++d) {
            row[d].index = static_cast<int>(d + 1);  // libsvm is 1-based
            row[d].value = static_cast<double>(train.X[i * args.dim + d]);
        }
        row[args.dim].index = -1;  // terminator
        row[args.dim].value = 0.0;
    }
    prob.y = y_train.data();
    prob.x = x_ptrs.data();

    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = args.gamma;
    param.coef0 = 0;
    param.cache_size = args.cache_mb;
    param.C = args.C;
    param.eps = args.eps;
    param.nr_weight = 0;
    param.weight_label = nullptr;
    param.weight = nullptr;
    param.nu = 0.5;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;

    const char* param_check = svm_check_parameter(&prob, &param);
    if (param_check) {
        std::cerr << "Parameter error: " << param_check << "\n";
        return 1;
    }

    std::cout << "Training RBF SVM with C=" << param.C
            << ", gamma=" << param.gamma
            << ", cache=" << param.cache_size << "MB, eps=" << param.eps << "\n";

    auto quiet = [](const char*) {};
    svm_set_print_string_function(quiet);

    auto t_train0 = std::chrono::high_resolution_clock::now();
    svm_model* model = svm_train(&prob, &param);
    auto t_train1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> dt_train = t_train1 - t_train0;

    auto evaluate = [&](const DatasetMem& ds, const std::string& name) {
        std::vector<int> conf(num_classes * num_classes, 0);
        std::size_t correct = 0;
        std::vector<svm_node> nodes(ds.n * (args.dim + 1));
        for (std::size_t i = 0; i < ds.n; ++i) {
            svm_node* row = &nodes[i * (args.dim + 1)];
            for (std::size_t d = 0; d < args.dim; ++d) {
                row[d].index = static_cast<int>(d + 1);
                row[d].value = static_cast<double>(ds.X[i * args.dim + d]);
            }
            row[args.dim].index = -1;
            row[args.dim].value = 0.0;

            int pred = static_cast<int>(svm_predict(model, row));
            int y = ds.y[i];
            if (pred == y) correct++;
            conf[y * num_classes + pred] += 1;
        }
        double acc = static_cast<double>(correct) / static_cast<double>(ds.n);
        std::cout << name << " accuracy: " << acc * 100.0 << "% (" << correct
                  << "/" << ds.n << ")\n";
        std::cout << "Confusion matrix (rows=true, cols=pred):\n";
        for (int r = 0; r < num_classes; ++r) {
            for (int c = 0; c < num_classes; ++c) {
                std::cout << conf[r * num_classes + c] << (c + 1 == num_classes ? "" : " ");
            }
            std::cout << "\n";
        }
    };

    auto t_eval0 = std::chrono::high_resolution_clock::now();
    evaluate(train, "Train");
    evaluate(test, "Test");
    auto t_eval1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt_eval = t_eval1 - t_eval0;

    std::cout << "Timing: load=" << dt_load.count() << "s, train=" << dt_train.count()
              << "s, eval=" << dt_eval.count() << "s\n";

    svm_free_and_destroy_model(&model);
    return 0;
}
