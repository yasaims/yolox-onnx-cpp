#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "postprocess/detection.hpp"

namespace yolox::io {

// class_id から決定的に色を決める。同じクラスは常に同じ色になる。
cv::Scalar ColorForClass(int class_id);

// 検出結果を image に直接描画する (矩形 + "label 0.87" のラベル背景付き)。
// class_id が kCocoClassNames の範囲外の場合は "class_<id>" にフォールバックする。
void DrawDetections(cv::Mat& image, const std::vector<postprocess::Detection>& detections);

}  // namespace yolox::io
