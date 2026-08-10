// Phase 2 CLI entry point: letterbox前処理 -> 推論 -> デコード/NMS -> 座標逆変換 ->
// 描画、までの一気通貫パイプライン。推論コア (engine/) は前後処理の知識を持たず、
// 本ファイルが各モジュールを接続する。
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "engine/inference_session.hpp"
#include "io/draw.hpp"
#include "postprocess/decode.hpp"
#include "postprocess/detection.hpp"
#include "postprocess/nms.hpp"
#include "preprocess/letterbox.hpp"

namespace {

struct CliArgs {
    std::string model_path;
    std::string input_path;
    std::string output_path = "output.jpg";
    int size = 416;
    float score_threshold = 0.30F;
    float nms_threshold = 0.45F;
    bool verbose = false;
};

void PrintUsage(const char* program_name) {
    std::cerr << "Usage: " << program_name
              << " --model <model.onnx> --input <image> [--output result.jpg]"
                 " [--size 416] [--score-thr 0.30] [--nms-thr 0.45] [--verbose]\n";
}

std::optional<CliArgs> ParseArgs(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&]() -> std::optional<std::string> {
            if (i + 1 >= argc) return std::nullopt;
            return std::string(argv[++i]);
        };

        if (arg == "--model") {
            auto value = next_value();
            if (!value) return std::nullopt;
            args.model_path = *value;
        } else if (arg == "--input") {
            auto value = next_value();
            if (!value) return std::nullopt;
            args.input_path = *value;
        } else if (arg == "--output") {
            auto value = next_value();
            if (!value) return std::nullopt;
            args.output_path = *value;
        } else if (arg == "--size") {
            auto value = next_value();
            if (!value) return std::nullopt;
            args.size = std::atoi(value->c_str());
        } else if (arg == "--score-thr") {
            auto value = next_value();
            if (!value) return std::nullopt;
            args.score_threshold = std::strtof(value->c_str(), nullptr);
        } else if (arg == "--nms-thr") {
            auto value = next_value();
            if (!value) return std::nullopt;
            args.nms_threshold = std::strtof(value->c_str(), nullptr);
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return std::nullopt;
        }
    }
    if (args.model_path.empty() || args.input_path.empty() || args.size <= 0) {
        return std::nullopt;
    }
    if (args.score_threshold < 0.0F || args.score_threshold > 1.0F || args.nms_threshold < 0.0F ||
        args.nms_threshold > 1.0F) {
        std::cerr << "--score-thr / --nms-thr must be within [0.0, 1.0]\n";
        return std::nullopt;
    }
    return args;
}

void PrintTensorInfo(const char* label, const yolox::engine::TensorInfo& info) {
    std::cout << label << " \"" << info.name << "\" shape=[";
    for (size_t i = 0; i < info.shape.size(); ++i) {
        std::cout << info.shape[i];
        if (i + 1 < info.shape.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    const auto args = ParseArgs(argc, argv);
    if (!args) {
        PrintUsage(argv[0]);
        return 1;
    }

    const cv::Mat image = cv::imread(args->input_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to read input image: " << args->input_path << "\n";
        return 2;
    }

    std::unique_ptr<yolox::engine::InferenceSession> session;
    try {
        session = std::make_unique<yolox::engine::InferenceSession>(args->model_path);
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load model \"" << args->model_path << "\": " << e.what() << "\n";
        return 2;
    }

    if (args->verbose) {
        for (const auto& info : session->inputs()) PrintTensorInfo("input ", info);
        for (const auto& info : session->outputs()) PrintTensorInfo("output", info);
    }

    const cv::Size target_size(args->size, args->size);
    const yolox::preprocess::LetterboxResult letterboxed =
        yolox::preprocess::Letterbox(image, target_size);
    const std::vector<float> input_data = yolox::preprocess::ToChwFloat(letterboxed.image);
    const std::vector<int64_t> input_shape = {1, 3, args->size, args->size};

    std::vector<Ort::Value> outputs;
    try {
        outputs = session->run(input_data, input_shape);
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference failed: " << e.what() << "\n";
        return 3;
    }

    if (outputs.empty()) {
        std::cerr << "Model produced no output tensors\n";
        return 4;
    }

    const auto& output_infos = session->outputs();
    if (args->verbose) {
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
            const size_t element_count = outputs[i].GetTensorTypeAndShapeInfo().GetElementCount();
            const std::string name =
                i < output_infos.size() ? output_infos[i].name : "output" + std::to_string(i);

            std::cout << name << " shape=[";
            for (size_t d = 0; d < shape.size(); ++d) {
                std::cout << shape[d];
                if (d + 1 < shape.size()) std::cout << ", ";
            }
            std::cout << "] elements=" << element_count << " first10=[";

            const float* data = outputs[i].GetTensorData<float>();
            const size_t preview_count = std::min<size_t>(10, element_count);
            for (size_t j = 0; j < preview_count; ++j) {
                std::cout << data[j];
                if (j + 1 < preview_count) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }

    // 検出ヘッド出力は shape [1, num_anchors, 5+num_classes] を想定する。
    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 3 || out_shape[0] != 1 || out_shape[2] <= 5) {
        std::cerr << "Unexpected output tensor shape (expected [1, num_anchors, 5+num_classes])\n";
        return 4;
    }
    const size_t num_anchors = static_cast<size_t>(out_shape[1]);
    const size_t num_attrs = static_cast<size_t>(out_shape[2]);
    const float* output_data = outputs[0].GetTensorData<float>();

    std::vector<yolox::postprocess::Detection> detections;
    try {
        yolox::postprocess::DecodeConfig decode_config;
        decode_config.input_size = target_size;
        decode_config.score_threshold = args->score_threshold;
        detections = yolox::postprocess::Decode(output_data, num_anchors, num_attrs, decode_config);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Decode failed: " << e.what() << "\n";
        return 4;
    }

    detections = yolox::postprocess::NonMaxSuppression(detections, args->nms_threshold);

    for (auto& det : detections) {
        det = yolox::postprocess::ToOriginalScale(det, letterboxed.info.ratio, image.size());
    }

    std::cout << "detections=" << detections.size() << "\n";
    for (const auto& det : detections) {
        std::cout << "  class=" << det.class_id << " score=" << det.score << " x=" << det.box.x
                   << " y=" << det.box.y << " w=" << det.box.width << " h=" << det.box.height << "\n";
    }

    cv::Mat output_image = image.clone();
    yolox::io::DrawDetections(output_image, detections);

    if (!cv::imwrite(args->output_path, output_image)) {
        std::cerr << "Failed to write output image: " << args->output_path << "\n";
        return 5;
    }

    return 0;
}
