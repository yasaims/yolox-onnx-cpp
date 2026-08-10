// Phase 1 CLI entry point: prove the ONNX Runtime + OpenCV pipeline links
// and runs end-to-end. Pre/post-processing here is intentionally minimal —
// letterbox resizing, decoding, NMS, and drawing are Phase 2 work.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "engine/inference_session.hpp"

namespace {

struct CliArgs {
    std::string model_path;
    std::string input_path;
    int size = 416;
    bool verbose = false;
};

void PrintUsage(const char* program_name) {
    std::cerr << "Usage: " << program_name
              << " --model <model.onnx> --input <image> [--size 416] [--verbose]\n";
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
        } else if (arg == "--size") {
            auto value = next_value();
            if (!value) return std::nullopt;
            args.size = std::atoi(value->c_str());
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
    return args;
}

// TODO(phase2): replace with aspect-ratio-preserving letterbox resize in
// src/preprocess/. YOLOX also expects raw 0-255 pixel values (no /255
// normalization) — that part of this stub already matches the final
// pipeline and can stay.
std::vector<float> PreprocessImage(const cv::Mat& bgr_image, int size) {
    cv::Mat resized;
    cv::resize(bgr_image, resized, cv::Size(size, size));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F);

    // HWC -> CHW
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> chw(static_cast<size_t>(3) * size * size);
    const size_t plane_size = static_cast<size_t>(size) * size;
    for (int c = 0; c < 3; ++c) {
        std::memcpy(chw.data() + c * plane_size, channels[c].ptr<float>(), plane_size * sizeof(float));
    }
    return chw;
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

    const std::vector<float> input_data = PreprocessImage(image, args->size);
    const std::vector<int64_t> input_shape = {1, 3, args->size, args->size};

    std::vector<Ort::Value> outputs;
    try {
        outputs = session->run(input_data, input_shape);
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference failed: " << e.what() << "\n";
        return 3;
    }

    const auto& output_infos = session->outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
        const size_t element_count = outputs[i].GetTensorTypeAndShapeInfo().GetElementCount();
        const std::string name = i < output_infos.size() ? output_infos[i].name : "output" + std::to_string(i);

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

    return 0;
}
