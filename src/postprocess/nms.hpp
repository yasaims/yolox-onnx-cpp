#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "postprocess/detection.hpp"

namespace yolox::postprocess {

float IoU(const cv::Rect2f& a, const cv::Rect2f& b);

// クラス別 greedy NMS。
std::vector<Detection> NonMaxSuppression(const std::vector<Detection>& detections,
                                          float iou_threshold);

}  // namespace yolox::postprocess
