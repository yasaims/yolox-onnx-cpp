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

class DetectorError : public std::runtime_error {
public:
    explicit DetectorError(const std::string& what) : std::runtime_error(what) {}
};

// 既定引数は使わずオーバーロードで代替する (GCC 16.1.0 のバグ回避)。
class Detector {
public:
    Detector(const std::string& model_path, const DetectorConfig& config);
    explicit Detector(const std::string& model_path) : Detector(model_path, DetectorConfig{}) {}

    std::vector<postprocess::Detection> Detect(const cv::Mat& image);

    const DetectorConfig& config() const noexcept { return config_; }
    const engine::InferenceSession& session() const noexcept { return session_; }

private:
    engine::InferenceSession session_;
    DetectorConfig config_;
};

}  // namespace yolox::pipeline
