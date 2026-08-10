#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace yolox::preprocess {

// letterbox の変換パラメータ。後処理の座標逆変換がこれだけを必要とする。
// YOLOX はパディングを右下にのみ付けるため、オフセットは常に 0 —
// 記録する必要があるのは倍率 1 つだけ。
struct LetterboxInfo {
    float ratio = 1.0F;
};

struct LetterboxResult {
    cv::Mat image;  // CV_8UC3, 常に target と同じ大きさ
    LetterboxInfo info;
};

// アスペクト比を保ったまま target 内に収め、余白を pad_value で右下に埋める。
// 既定引数は使わない (GCC 16.1.0 のバグ回避、CLAUDE.md参照) — オーバーロードで代替する。
LetterboxResult Letterbox(const cv::Mat& src, const cv::Size& target, int pad_value);
LetterboxResult Letterbox(const cv::Mat& src, const cv::Size& target);  // pad_value = 114

// HWC uint8 BGR -> CHW float32 (0-255スケールのまま、色順変換なし)。
std::vector<float> ToChwFloat(const cv::Mat& hwc_bgr);

}  // namespace yolox::preprocess
