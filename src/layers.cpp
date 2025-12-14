#include "layers.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace {
inline std::size_t idx4(std::size_t n, std::size_t c, std::size_t h,
                        std::size_t w, std::size_t C, std::size_t H,
                        std::size_t W) {
    return ((n * C + c) * H + h) * W + w;
}

inline std::size_t weight_idx(std::size_t oc, std::size_t ic, std::size_t kh,
                              std::size_t kw, std::size_t in_c) {
    return ((oc * in_c + ic) * 3 + kh) * 3 + kw;
}
}  // namespace

Conv2D::Conv2D(std::size_t in_channels, std::size_t out_channels)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      weights_(out_channels * in_channels * 3 * 3, 0.0f),
      bias_(out_channels, 0.0f),
      grad_weights_(out_channels * in_channels * 3 * 3, 0.0f),
      grad_bias_(out_channels, 0.0f) {}

std::vector<float> Conv2D::forward(const std::vector<float>& input,
                                   std::size_t batch, std::size_t height,
                                   std::size_t width) {
    const std::size_t out_h = height;
    const std::size_t out_w = width;
    std::vector<float> output(batch * out_channels_ * out_h * out_w, 0.0f);

    input_cache_ = input;
    cached_batch_ = batch;
    cached_h_ = height;
    cached_w_ = width;

    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t oc = 0; oc < out_channels_; ++oc) {
            for (std::size_t h = 0; h < out_h; ++h) {
                for (std::size_t w = 0; w < out_w; ++w) {
                    float acc = bias_[oc];
                    for (std::size_t ic = 0; ic < in_channels_; ++ic) {
                        for (std::size_t kh = 0; kh < 3; ++kh) {
                            for (std::size_t kw = 0; kw < 3; ++kw) {
                                const int in_h = static_cast<int>(h) + static_cast<int>(kh) - 1;
                                const int in_w = static_cast<int>(w) + static_cast<int>(kw) - 1;
                                if (in_h < 0 || in_w < 0 ||
                                    in_h >= static_cast<int>(height) ||
                                    in_w >= static_cast<int>(width)) {
                                    continue;  // padding=0 outside
                                }
                                const std::size_t in_idx =
                                    idx4(n, ic, static_cast<std::size_t>(in_h),
                                         static_cast<std::size_t>(in_w),
                                         in_channels_, height, width);
                                const std::size_t w_idx =
                                    weight_idx(oc, ic, kh, kw, in_channels_);
                                acc += input[in_idx] * weights_[w_idx];
                            }
                        }
                    }
                    output[idx4(n, oc, h, w, out_channels_, out_h, out_w)] = acc;
                }
            }
        }
    }
    return output;
}

std::vector<float> Conv2D::backward(const std::vector<float>& grad_output,
                                    std::size_t batch, std::size_t height,
                                    std::size_t width) {
    if (input_cache_.empty()) {
        throw std::runtime_error("Conv2D backward called without forward cache");
    }
    if (batch != cached_batch_ || height != cached_h_ || width != cached_w_) {
        throw std::runtime_error("Conv2D backward shape mismatch with cache");
    }

    std::vector<float> grad_input(batch * in_channels_ * height * width, 0.0f);

    // Grad bias: sum over N, H, W.
    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t oc = 0; oc < out_channels_; ++oc) {
            float acc = 0.0f;
            for (std::size_t h = 0; h < height; ++h) {
                for (std::size_t w = 0; w < width; ++w) {
                    acc += grad_output[idx4(n, oc, h, w, out_channels_, height, width)];
                }
            }
            grad_bias_[oc] += acc;
        }
    }

    // Grad weights and grad input.
    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t oc = 0; oc < out_channels_; ++oc) {
            for (std::size_t h = 0; h < height; ++h) {
                for (std::size_t w = 0; w < width; ++w) {
                    const float go =
                        grad_output[idx4(n, oc, h, w, out_channels_, height, width)];
                    if (go == 0.0f) continue;  // small skip when ReLU zeros.
                    for (std::size_t ic = 0; ic < in_channels_; ++ic) {
                        for (std::size_t kh = 0; kh < 3; ++kh) {
                            for (std::size_t kw = 0; kw < 3; ++kw) {
                                const int in_h = static_cast<int>(h) + static_cast<int>(kh) - 1;
                                const int in_w = static_cast<int>(w) + static_cast<int>(kw) - 1;
                                if (in_h < 0 || in_w < 0 ||
                                    in_h >= static_cast<int>(height) ||
                                    in_w >= static_cast<int>(width)) {
                                    continue;
                                }
                                const std::size_t in_idx =
                                    idx4(n, ic, static_cast<std::size_t>(in_h),
                                         static_cast<std::size_t>(in_w),
                                         in_channels_, height, width);
                                const std::size_t w_idx =
                                    weight_idx(oc, ic, kh, kw, in_channels_);
                                grad_weights_[w_idx] += input_cache_[in_idx] * go;
                                grad_input[in_idx] += weights_[w_idx] * go;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

void Conv2D::zero_grad() {
    std::fill(grad_weights_.begin(), grad_weights_.end(), 0.0f);
    std::fill(grad_bias_.begin(), grad_bias_.end(), 0.0f);
}

void Conv2D::step(float lr) {
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= lr * grad_weights_[i];
    }
    for (std::size_t i = 0; i < bias_.size(); ++i) {
        bias_[i] -= lr * grad_bias_[i];
    }
}

std::vector<float> ReLU::forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    mask_.resize(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (input[i] > 0.0f) {
            output[i] = input[i];
            mask_[i] = 1;
        } else {
            output[i] = 0.0f;
            mask_[i] = 0;
        }
    }
    return output;
}

std::vector<float> ReLU::backward(const std::vector<float>& grad_output) {
    if (mask_.empty() || mask_.size() != grad_output.size()) {
        throw std::runtime_error("ReLU backward called without forward cache");
    }
    std::vector<float> grad_input(grad_output.size());
    for (std::size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = mask_[i] ? grad_output[i] : 0.0f;
    }
    return grad_input;
}

std::vector<float> MaxPool2x2::forward(const std::vector<float>& input,
                                       std::size_t batch,
                                       std::size_t channels,
                                       std::size_t height,
                                       std::size_t width) {
    if (height % 2 != 0 || width % 2 != 0) {
        throw std::runtime_error("MaxPool2x2 expects even height and width");
    }
    const std::size_t out_h = height / 2;
    const std::size_t out_w = width / 2;
    std::vector<float> output(batch * channels * out_h * out_w, 0.0f);
    max_indices_.resize(output.size());

    cached_batch_ = batch;
    cached_channels_ = channels;
    cached_in_h_ = height;
    cached_in_w_ = width;

    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t oh = 0; oh < out_h; ++oh) {
                for (std::size_t ow = 0; ow < out_w; ++ow) {
                    const std::size_t h0 = oh * 2;
                    const std::size_t w0 = ow * 2;
                    float max_val = -std::numeric_limits<float>::infinity();
                    std::size_t max_idx = 0;
                    for (std::size_t kh = 0; kh < 2; ++kh) {
                        for (std::size_t kw = 0; kw < 2; ++kw) {
                            const std::size_t ih = h0 + kh;
                            const std::size_t iw = w0 + kw;
                            const std::size_t in_idx =
                                idx4(n, c, ih, iw, channels, height, width);
                            if (input[in_idx] > max_val) {
                                max_val = input[in_idx];
                                max_idx = in_idx;
                            }
                        }
                    }
                    const std::size_t out_idx =
                        idx4(n, c, oh, ow, channels, out_h, out_w);
                    output[out_idx] = max_val;
                    max_indices_[out_idx] = max_idx;
                }
            }
        }
    }
    return output;
}

std::vector<float> MaxPool2x2::backward(
    const std::vector<float>& grad_output) {
    if (max_indices_.empty()) {
        throw std::runtime_error("MaxPool2x2 backward called without forward cache");
    }
    const std::size_t out_h = cached_in_h_ / 2;
    const std::size_t out_w = cached_in_w_ / 2;
    if (grad_output.size() != max_indices_.size()) {
        throw std::runtime_error("MaxPool2x2 backward grad_output size mismatch");
    }

    std::vector<float> grad_input(cached_batch_ * cached_channels_ * cached_in_h_ *
                                      cached_in_w_,
                                  0.0f);

    for (std::size_t n = 0; n < cached_batch_; ++n) {
        for (std::size_t c = 0; c < cached_channels_; ++c) {
            for (std::size_t oh = 0; oh < out_h; ++oh) {
                for (std::size_t ow = 0; ow < out_w; ++ow) {
                    const std::size_t out_idx =
                        idx4(n, c, oh, ow, cached_channels_, out_h, out_w);
                    const std::size_t in_idx = max_indices_[out_idx];
                    grad_input[in_idx] += grad_output[out_idx];
                }
            }
        }
    }
    return grad_input;
}

std::vector<float> UpSample2x2::forward(const std::vector<float>& input,
                                        std::size_t batch,
                                        std::size_t channels,
                                        std::size_t height,
                                        std::size_t width) {
    const std::size_t out_h = height * 2;
    const std::size_t out_w = width * 2;
    std::vector<float> output(batch * channels * out_h * out_w, 0.0f);

    cached_batch_ = batch;
    cached_channels_ = channels;
    cached_in_h_ = height;
    cached_in_w_ = width;

    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t h = 0; h < height; ++h) {
                for (std::size_t w = 0; w < width; ++w) {
                    const float v = input[idx4(n, c, h, w, channels, height, width)];
                    const std::size_t oh = h * 2;
                    const std::size_t ow = w * 2;
                    output[idx4(n, c, oh, ow, channels, out_h, out_w)] = v;
                    output[idx4(n, c, oh + 1, ow, channels, out_h, out_w)] = v;
                    output[idx4(n, c, oh, ow + 1, channels, out_h, out_w)] = v;
                    output[idx4(n, c, oh + 1, ow + 1, channels, out_h, out_w)] = v;
                }
            }
        }
    }
    return output;
}

std::vector<float> UpSample2x2::backward(
    const std::vector<float>& grad_output) {
    if (cached_batch_ == 0 || cached_channels_ == 0 || cached_in_h_ == 0 ||
        cached_in_w_ == 0) {
        throw std::runtime_error("UpSample2x2 backward called without forward cache");
    }
    const std::size_t out_h = cached_in_h_ * 2;
    const std::size_t out_w = cached_in_w_ * 2;
    if (grad_output.size() != cached_batch_ * cached_channels_ * out_h * out_w) {
        throw std::runtime_error("UpSample2x2 backward grad_output size mismatch");
    }

    std::vector<float> grad_input(cached_batch_ * cached_channels_ * cached_in_h_ *
                                      cached_in_w_,
                                  0.0f);

    for (std::size_t n = 0; n < cached_batch_; ++n) {
        for (std::size_t c = 0; c < cached_channels_; ++c) {
            for (std::size_t h = 0; h < cached_in_h_; ++h) {
                for (std::size_t w = 0; w < cached_in_w_; ++w) {
                    const std::size_t oh = h * 2;
                    const std::size_t ow = w * 2;
                    const float g00 =
                        grad_output[idx4(n, c, oh, ow, cached_channels_, out_h, out_w)];
                    const float g10 = grad_output[idx4(n, c, oh + 1, ow,
                                                       cached_channels_, out_h, out_w)];
                    const float g01 = grad_output[idx4(n, c, oh, ow + 1,
                                                       cached_channels_, out_h, out_w)];
                    const float g11 = grad_output[idx4(n, c, oh + 1, ow + 1,
                                                       cached_channels_, out_h, out_w)];
                    grad_input[idx4(n, c, h, w, cached_channels_, cached_in_h_,
                                    cached_in_w_)] = g00 + g10 + g01 + g11;
                }
            }
        }
    }

    return grad_input;
}

float MSELoss::forward(const std::vector<float>& pred,
                       const std::vector<float>& target) {
    if (pred.size() != target.size()) {
        throw std::runtime_error("MSELoss forward size mismatch");
    }
    const std::size_t n = pred.size();
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float diff = pred[i] - target[i];
        sum += diff * diff;
    }
    return sum / static_cast<float>(n);
}

std::vector<float> MSELoss::backward(const std::vector<float>& pred,
                                     const std::vector<float>& target) const {
    if (pred.size() != target.size()) {
        throw std::runtime_error("MSELoss backward size mismatch");
    }
    const std::size_t n = pred.size();
    std::vector<float> grad(n);
    const float scale = 2.0f / static_cast<float>(n);
    for (std::size_t i = 0; i < n; ++i) {
        grad[i] = (pred[i] - target[i]) * scale;
    }
    return grad;
}
