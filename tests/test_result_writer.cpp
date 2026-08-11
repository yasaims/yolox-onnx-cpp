#include "io/result_writer.hpp"

#include <regex>
#include <sstream>

#include <gtest/gtest.h>

#include "postprocess/detection.hpp"

using yolox::io::ResultMeta;
using yolox::io::WriteJson;
using yolox::io::WriteText;
using yolox::postprocess::Detection;

namespace {

// scripts/verify_parity.py の DETECTION_LINE_RE と同じパターン。テキスト出力の
// "class=.. score=.. x=.. y=.. w=.. h=.." トークン列・並び順が崩れると parity 検証が
// 壊れるため、ここで回帰を検知する (Phase 4 計画書 §7 参照)。
const std::regex kDetectionLineRe(
    R"(class=(-?\d+)\s+score=([\d.eE+-]+)\s+x=([\d.eE+-]+)\s+y=([\d.eE+-]+)\s+w=([\d.eE+-]+)\s+h=([\d.eE+-]+))");

std::vector<Detection> SampleDetections() {
    Detection a;
    a.class_id = 0;
    a.score = 0.671259F;
    a.box = cv::Rect2f(45.9994F, 59.1137F, 355.777F, 448.329F);

    Detection b;
    b.class_id = 2;
    b.score = 0.5F;
    b.box = cv::Rect2f(1.0F, 2.0F, 3.0F, 4.0F);

    return {a, b};
}

ResultMeta SampleMeta() {
    ResultMeta meta;
    meta.model_path = "yolox_nano.onnx";
    meta.input_path = "tests/data/test.jpg";
    meta.input_size = 416;
    meta.score_threshold = 0.3F;
    meta.nms_threshold = 0.45F;
    meta.image_size = cv::Size(512, 512);
    return meta;
}

}  // namespace

TEST(ResultWriter, TextMatchesParityRegexForEachDetection) {
    std::ostringstream oss;
    WriteText(oss, SampleDetections(), {"person", "bicycle", "car"}, SampleMeta());
    const std::string text = oss.str();

    auto begin = std::sregex_iterator(text.begin(), text.end(), kDetectionLineRe);
    const auto end = std::sregex_iterator();
    const long match_count = std::distance(begin, end);
    EXPECT_EQ(match_count, 2);
}

TEST(ResultWriter, TextIncludesDetectionsCountAndLabelSuffix) {
    std::ostringstream oss;
    WriteText(oss, SampleDetections(), {"person", "bicycle", "car"}, SampleMeta());
    const std::string text = oss.str();

    EXPECT_NE(text.find("detections=2"), std::string::npos);
    EXPECT_NE(text.find("label=person"), std::string::npos);
    EXPECT_NE(text.find("label=car"), std::string::npos);
}

TEST(ResultWriter, TextEmptyDetectionsStillPrintsCount) {
    std::ostringstream oss;
    WriteText(oss, {}, {"person"}, SampleMeta());
    EXPECT_EQ(oss.str(), "detections=0\n");
}

TEST(ResultWriter, TextVideoFrameAddsFrameFpsHeader) {
    std::ostringstream oss;
    ResultMeta meta = SampleMeta();
    meta.frame_index = 7;
    meta.fps = 12.5;
    WriteText(oss, {}, {}, meta);

    EXPECT_NE(oss.str().find("frame=7 fps=12.5"), std::string::npos);
}

TEST(ResultWriter, JsonEmptyDetectionsProducesEmptyArray) {
    std::ostringstream oss;
    WriteJson(oss, {}, {"person"}, SampleMeta());
    const std::string json = oss.str();

    EXPECT_NE(json.find("\"detections\":[]"), std::string::npos);
    EXPECT_NE(json.find("\"model\":\"yolox_nano.onnx\""), std::string::npos);
}

TEST(ResultWriter, JsonMultipleDetectionsIncludesLabelAndBox) {
    std::ostringstream oss;
    WriteJson(oss, SampleDetections(), {"person", "bicycle", "car"}, SampleMeta());
    const std::string json = oss.str();

    EXPECT_NE(json.find("\"class_id\":0"), std::string::npos);
    EXPECT_NE(json.find("\"label\":\"person\""), std::string::npos);
    EXPECT_NE(json.find("\"label\":\"car\""), std::string::npos);
    EXPECT_NE(json.find("\"box\":{\"x\":"), std::string::npos);
}

TEST(ResultWriter, JsonEscapesQuotesAndBackslashesInLabels) {
    Detection det;
    det.class_id = 0;
    det.score = 0.9F;
    det.box = cv::Rect2f(0.0F, 0.0F, 1.0F, 1.0F);

    std::ostringstream oss;
    WriteJson(oss, {det}, {"weird \"quote\" and \\backslash\\"}, SampleMeta());
    const std::string json = oss.str();

    EXPECT_NE(json.find(R"(weird \"quote\" and \\backslash\\)"), std::string::npos);
}

TEST(ResultWriter, JsonOutOfRangeClassIdFallsBackToClassPrefix) {
    Detection det;
    det.class_id = 42;
    det.score = 0.9F;
    det.box = cv::Rect2f(0.0F, 0.0F, 1.0F, 1.0F);

    std::ostringstream oss;
    WriteJson(oss, {det}, {"only_one"}, SampleMeta());
    EXPECT_NE(oss.str().find("\"label\":\"class_42\""), std::string::npos);
}

TEST(ResultWriter, JsonVideoFrameIncludesFrameAndFps) {
    std::ostringstream oss;
    ResultMeta meta = SampleMeta();
    meta.frame_index = 3;
    meta.fps = 24.0;
    WriteJson(oss, {}, {}, meta);
    const std::string json = oss.str();

    EXPECT_NE(json.find("\"frame\":3"), std::string::npos);
    EXPECT_NE(json.find("\"fps\":24"), std::string::npos);
}
