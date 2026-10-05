#pragma once

#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

#include "postprocess/detection.hpp"

namespace yolox::postprocess {

struct GridStride {
    int grid_x;
    int grid_y;
    int stride;
};

// stride 昇順、各 stride 内は y 外側 / x 内側 (YOLOX 公式の順)。
std::vector<GridStride> GenerateGridStrides(const cv::Size& input_size,
                                             const std::vector<int>& strides);

struct DecodeConfig {
    cv::Size input_size;
    float score_threshold;
};

// data: [num_anchors, 5 + num_classes]。objectness / class 確率は ONNX 側で sigmoid 済みの前提。
// 返すボックスは letterbox 座標系。
std::vector<Detection> Decode(const float* data, size_t num_anchors, size_t num_attrs,
                               const DecodeConfig& config);

// パディングは右下のみのため ratio の除算のみで逆変換し、元画像でクリップする。
Detection ToOriginalScale(const Detection& det, float ratio, const cv::Size& original_size);

}  // namespace yolox::postprocess
