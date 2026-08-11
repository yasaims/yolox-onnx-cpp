#include "pipeline/detector.hpp"

#include "postprocess/decode.hpp"
#include "postprocess/nms.hpp"
#include "preprocess/letterbox.hpp"

namespace yolox::pipeline {

Detector::Detector(const std::string& model_path, const DetectorConfig& config)
    : session_(model_path), config_(config) {}

std::vector<postprocess::Detection> Detector::Detect(const cv::Mat& image) {
    const cv::Size target_size(config_.input_size, config_.input_size);
    const preprocess::LetterboxResult letterboxed = preprocess::Letterbox(image, target_size);
    const std::vector<float> input_data = preprocess::ToChwFloat(letterboxed.image);
    const std::vector<int64_t> input_shape = {1, 3, config_.input_size, config_.input_size};

    const std::vector<Ort::Value> outputs = session_.run(input_data, input_shape);
    if (outputs.empty()) {
        throw DetectorError("Model produced no output tensors");
    }

    // 検出ヘッド出力は shape [1, num_anchors, 5+num_classes] を想定する。
    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 3 || out_shape[0] != 1 || out_shape[2] <= 5) {
        throw DetectorError("Unexpected output tensor shape (expected [1, num_anchors, 5+num_classes])");
    }
    const size_t num_anchors = static_cast<size_t>(out_shape[1]);
    const size_t num_attrs = static_cast<size_t>(out_shape[2]);
    const float* output_data = outputs[0].GetTensorData<float>();

    std::vector<postprocess::Detection> detections;
    try {
        postprocess::DecodeConfig decode_config;
        decode_config.input_size = target_size;
        decode_config.score_threshold = config_.score_threshold;
        detections = postprocess::Decode(output_data, num_anchors, num_attrs, decode_config);
    } catch (const std::invalid_argument& e) {
        throw DetectorError(std::string("Decode failed: ") + e.what());
    }

    detections = postprocess::NonMaxSuppression(detections, config_.nms_threshold);

    for (auto& det : detections) {
        det = postprocess::ToOriginalScale(det, letterboxed.info.ratio, image.size());
    }

    return detections;
}

}  // namespace yolox::pipeline
