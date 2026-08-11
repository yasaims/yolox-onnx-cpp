#pragma once

#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

#include "postprocess/detection.hpp"

namespace yolox::postprocess {

// 1 アンカーに対応する grid 座標と stride。生成ロジックを単体で検証できるよう
// デコードから独立した関数に切り出す。
struct GridStride {
    int grid_x;
    int grid_y;
    int stride;
};

// strides 昇順・各 stride 内は y が外側/x が内側の順で YOLOX 公式と同じ並びの
// アンカー一覧を生成する。
std::vector<GridStride> GenerateGridStrides(const cv::Size& input_size,
                                             const std::vector<int>& strides);

struct DecodeConfig {
    cv::Size input_size;
    float score_threshold;
};

// data は [num_anchors, num_attrs] のフラット配列 (num_attrs = 5 + num_classes)。
// objectness / class 確率は ONNX グラフ側で sigmoid 適用済みの前提。
// grid/stride デコードのみここで行い、score_threshold 未満は捨てる。
// 出力ボックスは letterbox 座標系のまま (元画像スケールへの逆変換は呼び出し側)。
// アンカー数が GenerateGridStrides の結果と一致しない場合は std::invalid_argument を投げる。
std::vector<Detection> Decode(const float* data, size_t num_anchors, size_t num_attrs,
                               const DecodeConfig& config);

// letterbox 座標系 -> 元画像座標系。YOLOX のパディングは右下のみなので
// オフセット減算は不要。元画像の矩形でクリップして返す。
Detection ToOriginalScale(const Detection& det, float ratio, const cv::Size& original_size);

}  // namespace yolox::postprocess
