#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "postprocess/detection.hpp"

namespace yolox::io {

// class_id から決定的に色を決める。同じクラスは常に同じ色になる。
cv::Scalar ColorForClass(int class_id);

// 検出結果を image に直接描画する (矩形 + "label 0.87" のラベル背景付き)。
// labels 版はカスタムラベル配列を使う (範囲外は "class_<id>" にフォールバック、labels.hpp 参照)。
// 引数なし版は COCO 80 既定 (DefaultCocoLabels()) に委譲する。
void DrawDetections(cv::Mat& image, const std::vector<postprocess::Detection>& detections,
                     const std::vector<std::string>& labels);
void DrawDetections(cv::Mat& image, const std::vector<postprocess::Detection>& detections);

// 左上に半透明背景付きで "FPS: 12.3" を描画する (動画パスのオーバーレイ用)。
void DrawFps(cv::Mat& image, double fps);

}  // namespace yolox::io
