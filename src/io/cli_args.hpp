#pragma once

#include <string>

namespace yolox::io {

enum class OutputFormat { kText, kJson };
enum class RunMode { kAuto, kImage, kVideo };

struct CliArgs {
    std::string model_path;
    std::string input_path;
    std::string output_path;  // 空なら main が既定値を補う
    std::string labels_path;  // 空なら COCO 80
    int size = 416;
    float score_threshold = 0.30F;
    float nms_threshold = 0.45F;
    bool verbose = false;
    bool no_draw = false;
    OutputFormat format = OutputFormat::kText;
    RunMode mode = RunMode::kAuto;
    int max_frames = 0;  // 0 = 無制限
};

std::string UsageText(const char* program_name);

struct ParseResult {
    enum class Status { kOk, kHelp, kError };
    Status status = Status::kError;
    CliArgs args;
    std::string error;
};

ParseResult ParseArgs(int argc, const char* const* argv);

}  // namespace yolox::io
