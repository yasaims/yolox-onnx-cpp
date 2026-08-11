#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "engine/inference_session.hpp"
#include "postprocess/detection.hpp"

namespace yolox::pipeline {

struct DetectorConfig {
    int input_size = 416;
    float score_threshold = 0.30F;
    float nms_threshold = 0.45F;
};

// Detect() 内で出力テンソルの形状不整合・デコード失敗を報告するための例外型。
// InferenceSession のコンストラクタ / run() は Ort::Exception を投げるため区別する。
class DetectorError : public std::runtime_error {
public:
    explicit DetectorError(const std::string& what) : std::runtime_error(what) {}
};

// letterbox 前処理 -> 推論 -> grid/stride デコード -> NMS -> 元画像スケール逆変換、
// までの一気通貫パイプラインを1画像/1フレーム単位でまとめたもの。
// main.cpp の静止画パスと動画パスがロジックを共有するために切り出した (Phase 4)。
// 既定引数は使わない (GCC 16.1.0 のバグ回避、CLAUDE.md参照) — オーバーロードで代替する。
class Detector {
public:
    Detector(const std::string& model_path, const DetectorConfig& config);
    explicit Detector(const std::string& model_path) : Detector(model_path, DetectorConfig{}) {}

    // 1画像分の推論。元画像座標系の Detection を返す。
    // 推論自体の失敗は Ort::Exception、出力形状不整合・デコード失敗は DetectorError を送出する。
    std::vector<postprocess::Detection> Detect(const cv::Mat& image);

    const DetectorConfig& config() const noexcept { return config_; }
    const engine::InferenceSession& session() const noexcept { return session_; }

private:
    engine::InferenceSession session_;
    DetectorConfig config_;
};

}  // namespace yolox::pipeline
