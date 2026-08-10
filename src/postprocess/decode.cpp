#include "postprocess/decode.hpp"

#include <cmath>
#include <stdexcept>

namespace yolox::postprocess {

std::vector<GridStride> GenerateGridStrides(const cv::Size& input_size,
                                             const std::vector<int>& strides) {
    std::vector<GridStride> grid_strides;
    for (const int stride : strides) {
        const int grid_w = input_size.width / stride;
        const int grid_h = input_size.height / stride;
        grid_strides.reserve(grid_strides.size() + static_cast<size_t>(grid_w) * static_cast<size_t>(grid_h));
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                grid_strides.push_back(GridStride{x, y, stride});
            }
        }
    }
    return grid_strides;
}

std::vector<Detection> Decode(const float* data, size_t num_anchors, size_t num_attrs,
                               const DecodeConfig& config) {
    static const std::vector<int> kStrides = {8, 16, 32};
    const std::vector<GridStride> grid_strides = GenerateGridStrides(config.input_size, kStrides);

    if (grid_strides.size() != num_anchors) {
        throw std::invalid_argument(
            "Decode: anchor count mismatch between model output and generated grid "
            "(model/--size unsupported combination)");
    }

    const size_t num_classes = num_attrs - 5;
    std::vector<Detection> detections;

    for (size_t i = 0; i < num_anchors; ++i) {
        const float* row = data + i * num_attrs;
        const GridStride& gs = grid_strides[i];

        const float cx = (row[0] + static_cast<float>(gs.grid_x)) * static_cast<float>(gs.stride);
        const float cy = (row[1] + static_cast<float>(gs.grid_y)) * static_cast<float>(gs.stride);
        const float w = std::exp(row[2]) * static_cast<float>(gs.stride);
        const float h = std::exp(row[3]) * static_cast<float>(gs.stride);
        const float objectness = row[4];

        int best_class = -1;
        float best_class_score = 0.0F;
        for (size_t c = 0; c < num_classes; ++c) {
            const float class_score = row[5 + c];
            if (class_score > best_class_score) {
                best_class_score = class_score;
                best_class = static_cast<int>(c);
            }
        }

        const float score = objectness * best_class_score;
        if (score < config.score_threshold) {
            continue;
        }

        Detection det;
        det.box = cv::Rect2f(cx - w / 2.0F, cy - h / 2.0F, w, h);
        det.class_id = best_class;
        det.score = score;
        detections.push_back(det);
    }

    return detections;
}

Detection ToOriginalScale(const Detection& det, float ratio, const cv::Size& original_size) {
    Detection result = det;
    result.box.x /= ratio;
    result.box.y /= ratio;
    result.box.width /= ratio;
    result.box.height /= ratio;

    const cv::Rect2f image_bounds(0.0F, 0.0F, static_cast<float>(original_size.width),
                                   static_cast<float>(original_size.height));
    result.box = result.box & image_bounds;
    return result;
}

}  // namespace yolox::postprocess
