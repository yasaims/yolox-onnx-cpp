#include "io/cli_args.hpp"

#include <cstdlib>
#include <optional>

namespace yolox::io {

std::string UsageText(const char* program_name) {
    return std::string("Usage: ") + program_name +
           " --model <model.onnx> --input <image|video> [--output result.jpg|result.mp4]"
           " [--size 416] [--score-thr 0.30] [--nms-thr 0.45] [--labels labels.txt]"
           " [--format text|json] [--mode auto|image|video] [--max-frames N] [--no-draw]"
           " [--verbose] [--help]\n";
}

namespace {

std::optional<OutputFormat> ParseFormat(const std::string& value) {
    if (value == "text") return OutputFormat::kText;
    if (value == "json") return OutputFormat::kJson;
    return std::nullopt;
}

std::optional<RunMode> ParseMode(const std::string& value) {
    if (value == "auto") return RunMode::kAuto;
    if (value == "image") return RunMode::kImage;
    if (value == "video") return RunMode::kVideo;
    return std::nullopt;
}

}  // namespace

ParseResult ParseArgs(int argc, const char* const* argv) {
    ParseResult result;
    CliArgs& args = result.args;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&]() -> std::optional<std::string> {
            if (i + 1 >= argc) return std::nullopt;
            return std::string(argv[++i]);
        };

        if (arg == "--help" || arg == "-h") {
            result.status = ParseResult::Status::kHelp;
            return result;
        } else if (arg == "--model") {
            auto value = next_value();
            if (!value) {
                result.error = "--model requires a value";
                return result;
            }
            args.model_path = *value;
        } else if (arg == "--input") {
            auto value = next_value();
            if (!value) {
                result.error = "--input requires a value";
                return result;
            }
            args.input_path = *value;
        } else if (arg == "--output") {
            auto value = next_value();
            if (!value) {
                result.error = "--output requires a value";
                return result;
            }
            args.output_path = *value;
        } else if (arg == "--labels") {
            auto value = next_value();
            if (!value) {
                result.error = "--labels requires a value";
                return result;
            }
            args.labels_path = *value;
        } else if (arg == "--size") {
            auto value = next_value();
            if (!value) {
                result.error = "--size requires a value";
                return result;
            }
            args.size = std::atoi(value->c_str());
        } else if (arg == "--score-thr") {
            auto value = next_value();
            if (!value) {
                result.error = "--score-thr requires a value";
                return result;
            }
            args.score_threshold = std::strtof(value->c_str(), nullptr);
        } else if (arg == "--nms-thr") {
            auto value = next_value();
            if (!value) {
                result.error = "--nms-thr requires a value";
                return result;
            }
            args.nms_threshold = std::strtof(value->c_str(), nullptr);
        } else if (arg == "--format") {
            auto value = next_value();
            if (!value) {
                result.error = "--format requires a value";
                return result;
            }
            auto parsed = ParseFormat(*value);
            if (!parsed) {
                result.error = "--format must be 'text' or 'json'";
                return result;
            }
            args.format = *parsed;
        } else if (arg == "--mode") {
            auto value = next_value();
            if (!value) {
                result.error = "--mode requires a value";
                return result;
            }
            auto parsed = ParseMode(*value);
            if (!parsed) {
                result.error = "--mode must be 'auto', 'image' or 'video'";
                return result;
            }
            args.mode = *parsed;
        } else if (arg == "--max-frames") {
            auto value = next_value();
            if (!value) {
                result.error = "--max-frames requires a value";
                return result;
            }
            args.max_frames = std::atoi(value->c_str());
        } else if (arg == "--no-draw") {
            args.no_draw = true;
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else {
            result.error = "Unknown argument: " + arg;
            return result;
        }
    }

    if (args.model_path.empty()) {
        result.error = "--model is required";
        return result;
    }
    if (args.input_path.empty()) {
        result.error = "--input is required";
        return result;
    }
    if (args.size <= 0) {
        result.error = "--size must be a positive integer";
        return result;
    }
    if (args.score_threshold < 0.0F || args.score_threshold > 1.0F || args.nms_threshold < 0.0F ||
        args.nms_threshold > 1.0F) {
        result.error = "--score-thr / --nms-thr must be within [0.0, 1.0]";
        return result;
    }
    if (args.max_frames < 0) {
        result.error = "--max-frames must not be negative";
        return result;
    }

    result.status = ParseResult::Status::kOk;
    return result;
}

}  // namespace yolox::io
