#include "preprocess/letterbox.hpp"

#include <algorithm>
#include <cstring>

#include <opencv2/imgproc.hpp>

namespace yolox::preprocess {

LetterboxResult Letterbox(const cv::Mat& src, const cv::Size& target, int pad_value) {
    // パディングは右下のみ (中央寄せしない)。逆変換を ratio の除算だけで済ませるため。
    const float ratio = std::min(static_cast<float>(target.height) / static_cast<float>(src.rows),
                                  static_cast<float>(target.width) / static_cast<float>(src.cols));

    const int resized_w = std::max(1, static_cast<int>(static_cast<float>(src.cols) * ratio));
    const int resized_h = std::max(1, static_cast<int>(static_cast<float>(src.rows) * ratio));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat padded(target, src.type(), cv::Scalar(pad_value, pad_value, pad_value));
    resized.copyTo(padded(cv::Rect(0, 0, resized_w, resized_h)));

    LetterboxResult result;
    result.image = padded;
    result.info.ratio = ratio;
    return result;
}

LetterboxResult Letterbox(const cv::Mat& src, const cv::Size& target) {
    return Letterbox(src, target, 114);
}

std::vector<float> ToChwFloat(const cv::Mat& hwc_bgr) {
    cv::Mat float_img;
    hwc_bgr.convertTo(float_img, CV_32F);

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    const int height = hwc_bgr.rows;
    const int width = hwc_bgr.cols;
    const size_t plane_size = static_cast<size_t>(height) * static_cast<size_t>(width);

    std::vector<float> chw(plane_size * 3);
    for (int c = 0; c < 3; ++c) {
        // cv::split の出力は連続領域。
        std::memcpy(chw.data() + static_cast<size_t>(c) * plane_size, channels[c].ptr<float>(),
                    plane_size * sizeof(float));
    }
    return chw;
}

}  // namespace yolox::preprocess
