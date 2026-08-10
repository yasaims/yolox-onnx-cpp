#include "postprocess/decode.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

using yolox::postprocess::Decode;
using yolox::postprocess::DecodeConfig;
using yolox::postprocess::Detection;
using yolox::postprocess::GenerateGridStrides;
using yolox::postprocess::GridStride;
using yolox::postprocess::ToOriginalScale;

namespace {

// [num_anchors, num_attrs] のゼロ埋めバッファを作る。objectness/クラス確率を
// 明示的に置かない行は score=0 として自然にフィルタされる。
std::vector<float> MakeAnchorBuffer(size_t num_anchors, size_t num_attrs) {
    return std::vector<float>(num_anchors * num_attrs, 0.0F);
}

// index行に (raw_x, raw_y, raw_w, raw_h, objectness) と、指定クラスのスコアを書き込む。
void SetRow(std::vector<float>& buf, size_t num_attrs, size_t index, float raw_x, float raw_y,
            float raw_w, float raw_h, float objectness, int class_id, float class_score) {
    float* row = buf.data() + index * num_attrs;
    row[0] = raw_x;
    row[1] = raw_y;
    row[2] = raw_w;
    row[3] = raw_h;
    row[4] = objectness;
    row[5 + static_cast<size_t>(class_id)] = class_score;
}

}  // namespace

// --- GenerateGridStrides ---------------------------------------------------

TEST(GenerateGridStrides, AnchorCountMatchesSumOfGrids) {
    // 416: 52^2 + 26^2 + 13^2 = 2704 + 676 + 169 = 3549
    const auto grids_416 = GenerateGridStrides(cv::Size(416, 416), {8, 16, 32});
    EXPECT_EQ(grids_416.size(), 3549U);

    // 640: 80^2 + 40^2 + 20^2 = 6400 + 1600 + 400 = 8400
    const auto grids_640 = GenerateGridStrides(cv::Size(640, 640), {8, 16, 32});
    EXPECT_EQ(grids_640.size(), 8400U);
}

TEST(GenerateGridStrides, OrderIsStrideAscending) {
    // 416入力: stride8が52*52=2704件、続けてstride16が26*26=676件。
    const auto grids = GenerateGridStrides(cv::Size(416, 416), {8, 16, 32});
    EXPECT_EQ(grids[0].stride, 8);
    EXPECT_EQ(grids[2703].stride, 8);
    EXPECT_EQ(grids[2704].stride, 16);
    EXPECT_EQ(grids[3380].stride, 32);  // 2704 + 676 = 3380
}

TEST(GenerateGridStrides, WithinStrideYIsOuterXIsInner) {
    // 416入力ではstride8のgrid幅は52。yが外側・xが内側で並ぶため、
    // index 1はx方向に1つ進み、index 52はy方向に1つ進む。
    const auto grids = GenerateGridStrides(cv::Size(416, 416), {8, 16, 32});
    EXPECT_EQ(grids[0].grid_x, 0);
    EXPECT_EQ(grids[0].grid_y, 0);
    EXPECT_EQ(grids[1].grid_x, 1);
    EXPECT_EQ(grids[1].grid_y, 0);
    EXPECT_EQ(grids[52].grid_x, 0);
    EXPECT_EQ(grids[52].grid_y, 1);
}

// --- Decode ------------------------------------------------------------------

TEST(Decode, CenterAndSizeMath) {
    // 416入力・index 0 (grid 0,0 / stride 8)。raw_w=raw_h=0 -> exp(0)=1 -> w=h=stride。
    const size_t num_attrs = 85;  // 5 + 80クラス
    auto buf = MakeAnchorBuffer(3549, num_attrs);
    SetRow(buf, num_attrs, 0, /*raw_x=*/0.5F, /*raw_y=*/0.25F, /*raw_w=*/0.0F, /*raw_h=*/0.0F,
           /*objectness=*/1.0F, /*class_id=*/3, /*class_score=*/1.0F);

    DecodeConfig config;
    config.input_size = cv::Size(416, 416);
    config.score_threshold = 0.5F;
    const auto detections = Decode(buf.data(), 3549, num_attrs, config);

    ASSERT_EQ(detections.size(), 1U);
    const Detection& det = detections[0];
    // cx=(0.5+0)*8=4, cy=(0.25+0)*8=2, w=h=exp(0)*8=8 -> box=(cx-w/2, cy-h/2, w, h)
    EXPECT_FLOAT_EQ(det.box.x, 0.0F);
    EXPECT_FLOAT_EQ(det.box.y, -2.0F);
    EXPECT_FLOAT_EQ(det.box.width, 8.0F);
    EXPECT_FLOAT_EQ(det.box.height, 8.0F);
    EXPECT_EQ(det.class_id, 3);
    // letterbox座標系のまま返る (元画像への逆変換はToOriginalScaleの責務)。
}

TEST(Decode, AppliesGridOffsetForNonZeroAnchor) {
    // stride16の先頭はindex 2704 (grid 0,0)、続くindex 2705はgrid (1,0)。
    // cxがstride分(=16)だけ動くはずで、grid順序の仕様を実質的にピン留めする。
    const size_t num_attrs = 85;
    auto buf = MakeAnchorBuffer(3549, num_attrs);
    SetRow(buf, num_attrs, 2704, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0, 1.0F);
    SetRow(buf, num_attrs, 2705, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0, 1.0F);

    DecodeConfig config;
    config.input_size = cv::Size(416, 416);
    config.score_threshold = 0.5F;
    const auto detections = Decode(buf.data(), 3549, num_attrs, config);

    ASSERT_EQ(detections.size(), 2U);
    // cx = (0 + grid_x) * 16
    const float first_cx = detections[0].box.x + detections[0].box.width / 2.0F;
    const float second_cx = detections[1].box.x + detections[1].box.width / 2.0F;
    EXPECT_FLOAT_EQ(second_cx - first_cx, 16.0F);
}

TEST(Decode, ScoreIsObjectnessTimesClassScore) {
    // score_threshold は他の全ゼロ行 (score=0) を弾ける値にする必要がある
    // (実装は score < threshold で判定するため threshold=0 だと score=0 の行も残ってしまう)。
    const size_t num_attrs = 85;
    auto buf = MakeAnchorBuffer(3549, num_attrs);
    SetRow(buf, num_attrs, 0, 0.0F, 0.0F, 0.0F, 0.0F, /*objectness=*/0.8F, 0, /*class_score=*/0.5F);

    DecodeConfig config;
    config.input_size = cv::Size(416, 416);
    config.score_threshold = 0.01F;
    const auto detections = Decode(buf.data(), 3549, num_attrs, config);

    ASSERT_EQ(detections.size(), 1U);
    EXPECT_NEAR(detections[0].score, 0.4F, 1e-6F);
}

TEST(Decode, PicksHighestClassScore) {
    const size_t num_attrs = 85;
    auto buf = MakeAnchorBuffer(3549, num_attrs);
    float* row = buf.data();
    row[4] = 1.0F;         // objectness
    row[5 + 1] = 0.2F;     // class 1
    row[5 + 7] = 0.9F;     // class 7 (最大)
    row[5 + 40] = 0.6F;    // class 40

    DecodeConfig config;
    config.input_size = cv::Size(416, 416);
    config.score_threshold = 0.01F;  // 他の全ゼロ行 (score=0) を除外する
    const auto detections = Decode(buf.data(), 3549, num_attrs, config);

    ASSERT_EQ(detections.size(), 1U);
    EXPECT_EQ(detections[0].class_id, 7);
    EXPECT_NEAR(detections[0].score, 0.9F, 1e-6F);
}

TEST(Decode, FiltersBelowScoreThreshold) {
    const size_t num_attrs = 85;
    auto buf = MakeAnchorBuffer(3549, num_attrs);
    SetRow(buf, num_attrs, 0, 0.0F, 0.0F, 0.0F, 0.0F, 0.8F, 0, 0.5F);  // score=0.4

    DecodeConfig config;
    config.input_size = cv::Size(416, 416);
    config.score_threshold = 0.5F;
    const auto detections = Decode(buf.data(), 3549, num_attrs, config);

    EXPECT_TRUE(detections.empty());
}

TEST(Decode, KeepsScoreEqualToThreshold) {
    // 実装は score < threshold で捨てる (境界の等号は残す) 現仕様をピン留めする。
    const size_t num_attrs = 85;
    auto buf = MakeAnchorBuffer(3549, num_attrs);
    SetRow(buf, num_attrs, 0, 0.0F, 0.0F, 0.0F, 0.0F, 0.8F, 0, 0.5F);  // score=0.4

    DecodeConfig config;
    config.input_size = cv::Size(416, 416);
    config.score_threshold = 0.4F;
    const auto detections = Decode(buf.data(), 3549, num_attrs, config);

    EXPECT_EQ(detections.size(), 1U);
}

TEST(Decode, ThrowsOnAnchorCountMismatch) {
    // num_anchorsがGenerateGridStridesの結果と一致しない場合
    // (--size と実モデルの不整合を想定)。
    const size_t num_attrs = 85;
    auto buf = MakeAnchorBuffer(100, num_attrs);

    DecodeConfig config;
    config.input_size = cv::Size(416, 416);
    config.score_threshold = 0.0F;
    EXPECT_THROW(Decode(buf.data(), 100, num_attrs, config), std::invalid_argument);
}

// --- ToOriginalScale -----------------------------------------------------

TEST(ToOriginalScale, DividesByRatio) {
    Detection det;
    det.box = cv::Rect2f(10.0F, 20.0F, 30.0F, 40.0F);
    det.class_id = 0;
    det.score = 0.9F;

    const Detection result = ToOriginalScale(det, 0.5F, cv::Size(1000, 1000));

    EXPECT_FLOAT_EQ(result.box.x, 20.0F);
    EXPECT_FLOAT_EQ(result.box.y, 40.0F);
    EXPECT_FLOAT_EQ(result.box.width, 60.0F);
    EXPECT_FLOAT_EQ(result.box.height, 80.0F);
}

TEST(ToOriginalScale, ClipsToImageBounds) {
    Detection det;
    det.box = cv::Rect2f(90.0F, 90.0F, 30.0F, 30.0F);  // 元画像は100x100 -> はみ出す
    det.class_id = 0;
    det.score = 0.9F;

    const Detection result = ToOriginalScale(det, 1.0F, cv::Size(100, 100));

    EXPECT_FLOAT_EQ(result.box.x, 90.0F);
    EXPECT_FLOAT_EQ(result.box.y, 90.0F);
    EXPECT_FLOAT_EQ(result.box.width, 10.0F);
    EXPECT_FLOAT_EQ(result.box.height, 10.0F);
}

TEST(ToOriginalScale, FullyOutsideBoxBecomesEmpty) {
    // cv::Rect2f の & 演算子は交差なしを全ゼロ矩形として返す現挙動を記録する。
    Detection det;
    det.box = cv::Rect2f(200.0F, 200.0F, 10.0F, 10.0F);
    det.class_id = 0;
    det.score = 0.9F;

    const Detection result = ToOriginalScale(det, 1.0F, cv::Size(100, 100));

    EXPECT_FLOAT_EQ(result.box.width, 0.0F);
    EXPECT_FLOAT_EQ(result.box.height, 0.0F);
}
