// Phase 4 CLI entry point: 引数解析 -> ラベル読込 -> Detector構築 -> 静止画/動画モード分岐、
// までを行う薄い層。letterbox前処理〜座標逆変換の実体は pipeline::Detector (Phase 2の
// パイプラインを再利用可能なクラスへ抽出したもの) が持ち、静止画・動画の両パスで共有する。
#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "io/cli_args.hpp"
#include "io/draw.hpp"
#include "io/labels.hpp"
#include "io/result_writer.hpp"
#include "pipeline/detector.hpp"

namespace {

using yolox::io::CliArgs;
using yolox::io::OutputFormat;
using yolox::io::ResultMeta;
using yolox::io::RunMode;

void PrintTensorInfo(const char* label, const yolox::engine::TensorInfo& info) {
    std::cout << label << " \"" << info.name << "\" shape=[";
    for (size_t i = 0; i < info.shape.size(); ++i) {
        std::cout << info.shape[i];
        if (i + 1 < info.shape.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

// 拡張子 (小文字化) から動画かどうかを推定する。--mode auto のときのみ使う。
bool LooksLikeVideo(const std::string& path) {
    static const std::vector<std::string> kVideoExtensions = {".mp4", ".avi", ".mov",
                                                                ".mkv", ".webm", ".m4v"};
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return std::find(kVideoExtensions.begin(), kVideoExtensions.end(), ext) != kVideoExtensions.end();
}

bool ResolveIsVideo(const CliArgs& args) {
    switch (args.mode) {
        case RunMode::kImage:
            return false;
        case RunMode::kVideo:
            return true;
        case RunMode::kAuto:
        default:
            return LooksLikeVideo(args.input_path);
    }
}

std::string DefaultOutputPath(bool is_video) { return is_video ? "output.mp4" : "output.jpg"; }

ResultMeta BuildMeta(const CliArgs& args, const cv::Size& image_size) {
    ResultMeta meta;
    meta.model_path = args.model_path;
    meta.input_path = args.input_path;
    meta.input_size = args.size;
    meta.score_threshold = args.score_threshold;
    meta.nms_threshold = args.nms_threshold;
    meta.image_size = image_size;
    return meta;
}

void WriteResult(std::ostream& os, const std::vector<yolox::postprocess::Detection>& detections,
                  const std::vector<std::string>& labels, const ResultMeta& meta, OutputFormat format) {
    if (format == OutputFormat::kJson) {
        yolox::io::WriteJson(os, detections, labels, meta);
    } else {
        yolox::io::WriteText(os, detections, labels, meta);
    }
}

int RunImage(const CliArgs& args, yolox::pipeline::Detector& detector,
             const std::vector<std::string>& labels) {
    const cv::Mat image = cv::imread(args.input_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to read input image: " << args.input_path << "\n";
        return 2;
    }

    if (args.verbose) {
        for (const auto& info : detector.session().inputs()) PrintTensorInfo("input ", info);
        for (const auto& info : detector.session().outputs()) PrintTensorInfo("output", info);
    }

    std::vector<yolox::postprocess::Detection> detections;
    try {
        detections = detector.Detect(image);
    } catch (const yolox::pipeline::DetectorError& e) {
        std::cerr << e.what() << "\n";
        return 4;
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference failed: " << e.what() << "\n";
        return 3;
    }

    const ResultMeta meta = BuildMeta(args, image.size());
    WriteResult(std::cout, detections, labels, meta, args.format);

    if (!args.no_draw) {
        cv::Mat output_image = image.clone();
        yolox::io::DrawDetections(output_image, detections, labels);
        const std::string output_path = args.output_path.empty() ? DefaultOutputPath(false) : args.output_path;
        if (!cv::imwrite(output_path, output_image)) {
            std::cerr << "Failed to write output image: " << output_path << "\n";
            return 5;
        }
    }

    return 0;
}

int RunVideo(const CliArgs& args, yolox::pipeline::Detector& detector,
             const std::vector<std::string>& labels) {
    cv::VideoCapture cap(args.input_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open input video: " << args.input_path << "\n";
        return 2;
    }

    if (args.verbose) {
        for (const auto& info : detector.session().inputs()) PrintTensorInfo("input ", info);
        for (const auto& info : detector.session().outputs()) PrintTensorInfo("output", info);
    }

    const std::string output_path = args.output_path.empty() ? DefaultOutputPath(true) : args.output_path;
    cv::VideoWriter writer;
    if (!args.no_draw) {
        const double input_fps = cap.get(cv::CAP_PROP_FPS);
        const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(output_path, fourcc, input_fps > 0 ? input_fps : 30.0, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "Failed to open output video for writing: " << output_path << "\n";
            return 5;
        }
    }

    std::deque<double> recent_frame_seconds;
    constexpr size_t kFpsWindow = 30;
    const auto run_start = std::chrono::steady_clock::now();

    int frame_index = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        if (args.max_frames > 0 && frame_index >= args.max_frames) break;

        const auto frame_start = std::chrono::steady_clock::now();

        std::vector<yolox::postprocess::Detection> detections;
        try {
            detections = detector.Detect(frame);
        } catch (const yolox::pipeline::DetectorError& e) {
            std::cerr << e.what() << "\n";
            return 4;
        } catch (const Ort::Exception& e) {
            std::cerr << "Inference failed: " << e.what() << "\n";
            return 3;
        }

        const auto frame_end = std::chrono::steady_clock::now();
        const double frame_seconds = std::chrono::duration<double>(frame_end - frame_start).count();
        recent_frame_seconds.push_back(frame_seconds);
        if (recent_frame_seconds.size() > kFpsWindow) recent_frame_seconds.pop_front();
        const double avg_seconds =
            std::accumulate(recent_frame_seconds.begin(), recent_frame_seconds.end(), 0.0) /
            static_cast<double>(recent_frame_seconds.size());
        const double fps = avg_seconds > 0.0 ? 1.0 / avg_seconds : 0.0;

        ResultMeta meta = BuildMeta(args, frame.size());
        meta.frame_index = frame_index;
        meta.fps = fps;
        WriteResult(std::cout, detections, labels, meta, args.format);

        if (!args.no_draw) {
            yolox::io::DrawDetections(frame, detections, labels);
            yolox::io::DrawFps(frame, fps);
            writer.write(frame);
        }

        ++frame_index;
    }

    const auto run_end = std::chrono::steady_clock::now();
    const double total_seconds = std::chrono::duration<double>(run_end - run_start).count();
    const double average_fps = total_seconds > 0.0 ? static_cast<double>(frame_index) / total_seconds : 0.0;
    std::cerr << "Processed " << frame_index << " frames in " << total_seconds
              << "s (average " << average_fps << " fps)\n";

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const auto parsed = yolox::io::ParseArgs(argc, argv);
    if (parsed.status == yolox::io::ParseResult::Status::kHelp) {
        std::cout << yolox::io::UsageText(argv[0]);
        return 0;
    }
    if (parsed.status == yolox::io::ParseResult::Status::kError) {
        std::cerr << parsed.error << "\n";
        std::cerr << yolox::io::UsageText(argv[0]);
        return 1;
    }
    const CliArgs& args = parsed.args;

    if (args.size % 32 != 0) {
        std::cerr << "Warning: --size " << args.size
                  << " is not a multiple of 32; YOLOX uses strides 8/16/32.\n";
    }

    std::vector<std::string> labels;
    try {
        labels = args.labels_path.empty() ? yolox::io::DefaultCocoLabels()
                                           : yolox::io::LoadLabels(args.labels_path);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }

    yolox::pipeline::DetectorConfig config;
    config.input_size = args.size;
    config.score_threshold = args.score_threshold;
    config.nms_threshold = args.nms_threshold;

    std::unique_ptr<yolox::pipeline::Detector> detector;
    try {
        detector = std::make_unique<yolox::pipeline::Detector>(args.model_path, config);
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load model \"" << args.model_path << "\": " << e.what() << "\n";
        return 2;
    }

    const bool is_video = ResolveIsVideo(args);
    return is_video ? RunVideo(args, *detector, labels) : RunImage(args, *detector, labels);
}
