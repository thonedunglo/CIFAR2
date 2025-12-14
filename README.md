# CIFAR Autoencoder - Build & Run Cheatsheet

Tong hop cac lenh bien dich/chay tuong ung voi pipeline hien tai. Mac dinh o thu muc `~/data/CIFAR2` (Linux) hoac `D:\CIFAR2` (Windows PowerShell). Thay doi duong dan neu moi truong khac.

---

## Sanity check (loader)

Build (g++ 11+):
`g++ -std=c++17 -O2 src/sanity_check.cpp src/dataset.cpp -o sanity_check`

Chạy (PowerShell tại `D:\CIFAR2`):
`.\sanity_check .\cifar-10-batches-bin`

Chạy (bash tại `~/data/CIFAR2`):
`./sanity_check ./cifar-10-batches-bin`

## Sanity check (Conv2D)

Build:
`g++ -std=c++17 -O2 src/sanity_conv.cpp src/layers.cpp -o sanity_conv`

Chạy:
`./sanity_conv`

## Sanity check (ReLU)

Build:
`g++ -std=c++17 -O2 src/sanity_relu.cpp src/layers.cpp -o sanity_relu`

Chạy:
`./sanity_relu`

## Sanity check (MaxPool 2x2)

Build:
`g++ -std=c++17 -O2 src/sanity_pool.cpp src/layers.cpp -o sanity_pool`

Chạy:
`./sanity_pool`

## Sanity check (UpSample 2x2)

Build:
`g++ -std=c++17 -O2 src/sanity_upsample.cpp src/layers.cpp -o sanity_upsample`

Chạy:
`./sanity_upsample`

## Sanity check (MSE Loss)

Build:
`g++ -std=c++17 -O2 src/sanity_mse.cpp src/layers.cpp -o sanity_mse`

Chạy:
`./sanity_mse`

## Sanity check (Autoencoder init)

Build:
`g++ -std=c++17 -O2 src/sanity_autoencoder_init.cpp src/autoencoder.cpp src/layers.cpp -o sanity_autoencoder_init`

Chạy:
`./sanity_autoencoder_init`

## Sanity check (Autoencoder forward shapes)

Build:
`g++ -std=c++17 -O2 src/sanity_autoencoder_forward.cpp src/autoencoder.cpp src/layers.cpp -o sanity_autoencoder_forward`

Chạy:
`./sanity_autoencoder_forward`

## Sanity check (Autoencoder backward)

Build:
`g++ -std=c++17 -O2 src/sanity_autoencoder_backward.cpp src/autoencoder.cpp src/layers.cpp -o sanity_autoencoder_backward`

Chạy:
`./sanity_autoencoder_backward`

## Sanity check (SGD update)

Build:
`g++ -std=c++17 -O2 src/sanity_sgd.cpp src/autoencoder.cpp src/layers.cpp -o sanity_sgd`

Chạy:
`./sanity_sgd`

## Sanity check (Feature extraction)

Build:
`g++ -std=c++17 -O2 src/sanity_extract_features.cpp src/autoencoder.cpp src/layers.cpp -o sanity_extract_features`

Chạy:
`./sanity_extract_features`

## Sanity check (Persistence save/load)

Build:
`g++ -std=c++17 -O2 src/sanity_persistence.cpp src/autoencoder.cpp src/layers.cpp -o sanity_persistence`

Chạy:
`./sanity_persistence`

## Train CPU baseline

Build:
`g++ -std=c++17 -O2 src/train_cpu.cpp src/autoencoder.cpp src/layers.cpp src/dataset.cpp -o train_cpu`

Chạy (PowerShell tại `D:\CIFAR2`):
`.\train_cpu .\cifar-10-batches-bin`

Chạy (bash tại `~/data/CIFAR2`):
`./train_cpu ./cifar-10-batches-bin`

## GPU memory sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_mem.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_mem`

Chạy:
`./sanity_gpu_mem`

## GPU conv forward sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_conv.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_conv`

Chạy:
`./sanity_gpu_conv`

## GPU ReLU sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_relu.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_relu`

Chạy:
`./sanity_gpu_relu`

## GPU MaxPool sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_maxpool.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_maxpool`

Chạy:
`./sanity_gpu_maxpool`

## GPU UpSample sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_upsample.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_upsample`

Chạy:
`./sanity_gpu_upsample`

## GPU MSE sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_mse.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_mse`

Chạy:
`./sanity_gpu_mse`

## GPU H2D copy sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_copy.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_copy`

Chạy:
`./sanity_gpu_copy`

## GPU forward pipeline sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_forward.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_forward`

Chạy:
`./sanity_gpu_forward`

## GPU ReLU backward sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_relu_backward.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_relu_backward`

Chạy:
`./sanity_gpu_relu_backward`

## GPU UpSample backward sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_upsample_backward.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_upsample_backward`

Chạy:
`./sanity_gpu_upsample_backward`

## GPU D2H + host loss sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_loss_copy.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_loss_copy`

Chạy:
`./sanity_gpu_loss_copy`

## GPU Conv backward grad sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_conv_backward.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_conv_backward`

Chạy:
`./sanity_gpu_conv_backward`

## GPU SGD update sanity (requires CUDA/nvcc)

Build:
`nvcc -O2 src/sanity_gpu_sgd.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp -o sanity_gpu_sgd`

Chạy:
`./sanity_gpu_sgd`

## Train GPU naive (requires CUDA/nvcc)

Build:
`nvcc -O2 src/train_gpu.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp src/dataset.cpp -o train_gpu`

Chạy (bash tại `~/data/CIFAR2`):
`./train_gpu ./cifar-10-batches-bin`

## Inference (reconstruct one test image, PPM output, GPU)

Build:
`nvcc -O2 src/infer.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp src/dataset.cpp -o infer`

Chạy:
`./infer ./cifar-10-batches-bin ae_checkpoint_gpu.bin 0 infer`
- args: [data_dir] [checkpoint] [index] [out_prefix]
- output: `<out_prefix>_orig.ppm`, `<out_prefix>_recon.ppm`

## Checkpoint load sanity

Build:
`g++ -std=c++17 -O2 src/sanity_load_checkpoint.cpp src/autoencoder.cpp src/layers.cpp -o sanity_load_checkpoint`

Chạy:
`./sanity_load_checkpoint ae_checkpoint.bin`  (hoặc đường dẫn checkpoint khác)

## Extract batch features sanity

Build:
`g++ -std=c++17 -O2 src/sanity_extract_batch.cpp src/autoencoder.cpp src/layers.cpp src/dataset.cpp -o sanity_extract_batch`

Chạy (mặc định train split, batch 4):
`./sanity_extract_batch ./cifar-10-batches-bin ae_checkpoint.bin 4 train`

## Extract full features (CPU)

Build:
`g++ -std=c++17 -O2 src/extract_features.cpp src/autoencoder.cpp src/layers.cpp src/dataset.cpp -o extract_features`

Chạy:
`./extract_features ./cifar-10-batches-bin ae_checkpoint.bin features 128 0 50 0`
  - args: [data_dir] [checkpoint] [out_prefix] [batch_size] [sample] [log_interval] [test_sample]
  - sample: 0 = full train; >0 = giới hạn mẫu train
  - test_sample: 0 = full test; >0 = giới hạn mẫu test
  - output: features_train_features.bin, features_train_labels.bin, features_test_features.bin, features_test_labels.bin

## Extract full features (GPU)

Build:
`nvcc -O2 src/extract_features_gpu.cu src/gpu_autoencoder.cu src/autoencoder.cpp src/layers.cpp src/dataset.cpp -o extract_features_gpu`

Chạy:
`./extract_features_gpu ./cifar-10-batches-bin ae_checkpoint_gpu.bin features 128 0 50 0`
  - args: [data_dir] [checkpoint] [out_prefix] [batch_size] [sample] [log_interval] [test_sample]
  - sample/test_sample: 0 = full; >0 = giới hạn mẫu
  - latent 8192/ảnh, ghi file giống bản CPU

## Train/Eval SVM (RBF, libsvm)

Build:
`g++ -std=c++17 -O2 src/train_svm.cpp third_party/libsvm/svm.cpp -o train_svm -Ithird_party/libsvm`

Chạy:
`./train_svm features 8192 0 0 10 0.000122 200 0.001`
  - args: [feature_prefix] [dim] [sample_train] [sample_test] [C] [gamma] [cache_mb] [eps]
  - prefix tương ứng với file: <prefix>_train_features.bin, <prefix>_train_labels.bin, <prefix>_test_features.bin, <prefix>_test_labels.bin
  - sample_*: 0 = full set; >0 = giới hạn số mẫu
  - gamma: nếu =0 thì mặc định 1/dim

## Train/Eval SVM (GPU, cuML Python)

Yêu cầu môi trường có cuML (RAPIDS) + cupy.

Chạy:
`python src/train_svm_gpu.py features 8192 0 0 10 0.000122 1e-3`
  - args: [feature_prefix] [dim] [sample_train] [sample_test] [C] [gamma] [tol]
  - gamma=0 => mặc định 1/dim
  - sample_*: 0 = full; >0 = giới hạn mẫu
