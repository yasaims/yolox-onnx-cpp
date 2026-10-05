#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace yolox::preprocess {

// パディングは右下のみなのでオフセットは不要、倍率だけ持つ。
struct LetterboxInfo {
    float ratio = 1.0F;
};

struct LetterboxResult {
    cv::Mat image;  // CV_8UC3, 常に target と同じ大きさ
    LetterboxInfo info;
};

// アスペクト比を保って target 内に収め、余白を右下に pad_value で埋める。
// 既定引数は GCC のバグ回避のため使わずオーバーロードにしている。
LetterboxResult Letterbox(const cv::Mat& src, const cv::Size& target, int pad_value);
LetterboxResult Letterbox(const cv::Mat& src, const cv::Size& target);  // pad_value = 114

// HWC uint8 BGR -> CHW float32 (0-255スケールのまま、色順変換なし)。
std::vector<float> ToChwFloat(const cv::Mat& hwc_bgr);

}  // namespace yolox::preprocess
