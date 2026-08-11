#include "io/draw.hpp"

#include <algorithm>
#include <sstream>

#include <opencv2/imgproc.hpp>

#include "io/labels.hpp"

namespace yolox::io {

cv::Scalar ColorForClass(int class_id) {
    // クラスIDから決定的にHSVの色相を割り当て、BGRへ変換する。80色を手書きしない。
    const int hue = (class_id * 180) / 80;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 200, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Vec3b pixel = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}

void DrawDetections(cv::Mat& image, const std::vector<postprocess::Detection>& detections,
                     const std::vector<std::string>& labels) {
    const int thickness = std::max(1, std::min(image.cols, image.rows) / 400);

    for (const auto& det : detections) {
        const cv::Scalar color = ColorForClass(det.class_id);
        const cv::Rect box(det.box);
        cv::rectangle(image, box, color, thickness);

        std::ostringstream oss;
        oss << LabelFor(labels, det.class_id) << " " << std::fixed;
        oss.precision(2);
        oss << det.score;
        const std::string text = oss.str();

        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        const int label_top = std::max(box.y - text_size.height - baseline, 0);
        const cv::Rect label_bg(box.x, label_top, text_size.width + 2, text_size.height + baseline + 2);
        cv::rectangle(image, label_bg, color, cv::FILLED);
        cv::putText(image, text, cv::Point(box.x + 1, label_top + text_size.height + 1),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}

void DrawDetections(cv::Mat& image, const std::vector<postprocess::Detection>& detections) {
    DrawDetections(image, detections, DefaultCocoLabels());
}

void DrawFps(cv::Mat& image, double fps) {
    std::ostringstream oss;
    oss << "FPS: " << std::fixed;
    oss.precision(1);
    oss << fps;
    const std::string text = oss.str();

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    const cv::Rect bg(0, 0, text_size.width + 12, text_size.height + baseline + 12);
    cv::Mat roi = image(bg & cv::Rect(0, 0, image.cols, image.rows));
    cv::Mat overlay(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(overlay, 0.5, roi, 0.5, 0.0, roi);

    cv::putText(image, text, cv::Point(6, text_size.height + 6), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

}  // namespace yolox::io
