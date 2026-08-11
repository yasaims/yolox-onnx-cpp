#include "io/cli_args.hpp"

#include <gtest/gtest.h>

using yolox::io::CliArgs;
using yolox::io::OutputFormat;
using yolox::io::ParseArgs;
using yolox::io::ParseResult;
using yolox::io::RunMode;

namespace {

// argv には NULL 終端ではなく argc を渡す想定なので、末尾の nullptr は不要。
ParseResult Parse(std::vector<const char*> argv) { return ParseArgs(static_cast<int>(argv.size()), argv.data()); }

}  // namespace

TEST(CliArgs, MinimalRequiredArgsUsesDefaults) {
    const auto result = Parse({"prog", "--model", "m.onnx", "--input", "in.jpg"});

    ASSERT_EQ(result.status, ParseResult::Status::kOk);
    EXPECT_EQ(result.args.model_path, "m.onnx");
    EXPECT_EQ(result.args.input_path, "in.jpg");
    EXPECT_EQ(result.args.size, 416);
    EXPECT_FLOAT_EQ(result.args.score_threshold, 0.30F);
    EXPECT_FLOAT_EQ(result.args.nms_threshold, 0.45F);
    EXPECT_FALSE(result.args.verbose);
    EXPECT_FALSE(result.args.no_draw);
    EXPECT_EQ(result.args.format, OutputFormat::kText);
    EXPECT_EQ(result.args.mode, RunMode::kAuto);
    EXPECT_EQ(result.args.max_frames, 0);
    EXPECT_TRUE(result.args.labels_path.empty());
    EXPECT_TRUE(result.args.output_path.empty());
}

TEST(CliArgs, AllFlagsParsed) {
    const auto result = Parse({"prog", "--model", "m.onnx", "--input", "in.mp4", "--output", "out.mp4",
                                "--size", "640", "--score-thr", "0.5", "--nms-thr", "0.6", "--labels",
                                "l.txt", "--format", "json", "--mode", "video", "--max-frames", "30",
                                "--no-draw", "--verbose"});

    ASSERT_EQ(result.status, ParseResult::Status::kOk);
    EXPECT_EQ(result.args.output_path, "out.mp4");
    EXPECT_EQ(result.args.size, 640);
    EXPECT_FLOAT_EQ(result.args.score_threshold, 0.5F);
    EXPECT_FLOAT_EQ(result.args.nms_threshold, 0.6F);
    EXPECT_EQ(result.args.labels_path, "l.txt");
    EXPECT_EQ(result.args.format, OutputFormat::kJson);
    EXPECT_EQ(result.args.mode, RunMode::kVideo);
    EXPECT_EQ(result.args.max_frames, 30);
    EXPECT_TRUE(result.args.no_draw);
    EXPECT_TRUE(result.args.verbose);
}

TEST(CliArgs, HelpFlagShortCircuits) {
    const auto result = Parse({"prog", "--help"});
    EXPECT_EQ(result.status, ParseResult::Status::kHelp);

    const auto result_short = Parse({"prog", "-h"});
    EXPECT_EQ(result_short.status, ParseResult::Status::kHelp);
}

TEST(CliArgs, MissingRequiredArgsIsError) {
    EXPECT_EQ(Parse({"prog"}).status, ParseResult::Status::kError);
    EXPECT_EQ(Parse({"prog", "--model", "m.onnx"}).status, ParseResult::Status::kError);
    EXPECT_EQ(Parse({"prog", "--input", "in.jpg"}).status, ParseResult::Status::kError);
}

TEST(CliArgs, FlagWithoutValueIsError) {
    const auto result = Parse({"prog", "--model"});
    EXPECT_EQ(result.status, ParseResult::Status::kError);
    EXPECT_FALSE(result.error.empty());
}

TEST(CliArgs, UnknownArgumentIsError) {
    const auto result = Parse({"prog", "--model", "m.onnx", "--input", "in.jpg", "--bogus"});
    EXPECT_EQ(result.status, ParseResult::Status::kError);
}

TEST(CliArgs, ThresholdOutOfRangeIsError) {
    EXPECT_EQ(Parse({"prog", "--model", "m.onnx", "--input", "in.jpg", "--score-thr", "1.5"}).status,
              ParseResult::Status::kError);
    EXPECT_EQ(Parse({"prog", "--model", "m.onnx", "--input", "in.jpg", "--nms-thr", "-0.1"}).status,
              ParseResult::Status::kError);
}

TEST(CliArgs, NonPositiveSizeIsError) {
    EXPECT_EQ(Parse({"prog", "--model", "m.onnx", "--input", "in.jpg", "--size", "0"}).status,
              ParseResult::Status::kError);
}

TEST(CliArgs, InvalidFormatIsError) {
    EXPECT_EQ(Parse({"prog", "--model", "m.onnx", "--input", "in.jpg", "--format", "xml"}).status,
              ParseResult::Status::kError);
}

TEST(CliArgs, InvalidModeIsError) {
    EXPECT_EQ(Parse({"prog", "--model", "m.onnx", "--input", "in.jpg", "--mode", "bogus"}).status,
              ParseResult::Status::kError);
}

TEST(CliArgs, NegativeMaxFramesIsError) {
    EXPECT_EQ(Parse({"prog", "--model", "m.onnx", "--input", "in.jpg", "--max-frames", "-1"}).status,
              ParseResult::Status::kError);
}
