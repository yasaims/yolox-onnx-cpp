#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "postprocess/detection.hpp"

namespace yolox::io {

// 同じ class_id は常に同じ色。
cv::Scalar ColorForClass(int class_id);

// 検出結果を image に描画する。labels 範囲外の id は "class_<id>"。
void DrawDetections(cv::Mat& image, const std::vector<postprocess::Detection>& detections,
                     const std::vector<std::string>& labels);
void DrawDetections(cv::Mat& image, const std::vector<postprocess::Detection>& detections);

// 左上に FPS を描画する。
void DrawFps(cv::Mat& image, double fps);

}  // namespace yolox::io
