#pragma once

#include <string>

namespace yolox::io {

enum class OutputFormat { kText, kJson };
enum class RunMode { kAuto, kImage, kVideo };

struct CliArgs {
    std::string model_path;
    std::string input_path;
    std::string output_path;  // 空なら main側でモードに応じた既定値 (output.jpg / output.mp4) を補う
    std::string labels_path;  // 空なら COCO 80 既定
    int size = 416;
    float score_threshold = 0.30F;
    float nms_threshold = 0.45F;
    bool verbose = false;
    bool no_draw = false;
    OutputFormat format = OutputFormat::kText;
    RunMode mode = RunMode::kAuto;
    int max_frames = 0;  // 0 = 無制限 (動画のみ意味を持つ)
};

// usage 文字列 (--help / 引数エラー時の両方で使う)。
std::string UsageText(const char* program_name);

struct ParseResult {
    enum class Status { kOk, kHelp, kError };
    Status status = Status::kError;
    CliArgs args;
    std::string error;  // kError のときのみ、人間可読の理由
};

// argv からの手書き解析。ParseResult::status で --help / 通常成功 / 失敗を区別する。
// テストからリテラル配列を渡せるよう argv は const char* const* を受け取る。
ParseResult ParseArgs(int argc, const char* const* argv);

}  // namespace yolox::io
