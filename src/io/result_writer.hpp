#pragma once

#include <ostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "postprocess/detection.hpp"

namespace yolox::io {

struct ResultMeta {
    std::string model_path;
    std::string input_path;
    int input_size = 0;
    float score_threshold = 0.0F;
    float nms_threshold = 0.0F;
    cv::Size image_size;
    int frame_index = -1;  // 動画のとき >= 0、静止画のとき -1
    double fps = 0.0;      // 動画のとき > 0
};

// 既存トークン列・並びは scripts/verify_parity.py の正規表現が依存するため変更しない。
// 動画 (frame_index >= 0) は先頭に "frame=<i> fps=<f>" 行を出す。
void WriteText(std::ostream& os, const std::vector<postprocess::Detection>& detections,
               const std::vector<std::string>& labels, const ResultMeta& meta);

void WriteJson(std::ostream& os, const std::vector<postprocess::Detection>& detections,
               const std::vector<std::string>& labels, const ResultMeta& meta);

}  // namespace yolox::io
