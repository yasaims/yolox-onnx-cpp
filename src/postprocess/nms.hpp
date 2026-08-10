#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "postprocess/detection.hpp"

namespace yolox::postprocess {

// 交差矩形が空、または合計面積が 0 の場合は 0.0 を返す。
float IoU(const cv::Rect2f& a, const cv::Rect2f& b);

// クラス別 greedy NMS (自前実装、cv::dnn::NMSBoxes は使わない — docs/adr/0004参照)。
// スコア降順に走査し、同一クラスで IoU が iou_threshold を超える後続候補を抑制する。
// 入力は変更しない。
std::vector<Detection> NonMaxSuppression(const std::vector<Detection>& detections,
                                          float iou_threshold);

}  // namespace yolox::postprocess
