#include "postprocess/nms.hpp"

#include <gtest/gtest.h>

using yolox::postprocess::Detection;
using yolox::postprocess::IoU;
using yolox::postprocess::NonMaxSuppression;

TEST(IoUTest, IdenticalBoxesReturnOne) {
    const cv::Rect2f box(0, 0, 10, 10);
    EXPECT_FLOAT_EQ(IoU(box, box), 1.0F);
}

TEST(IoUTest, NonOverlappingBoxesReturnZero) {
    const cv::Rect2f a(0, 0, 10, 10);
    const cv::Rect2f b(20, 20, 10, 10);
    EXPECT_FLOAT_EQ(IoU(a, b), 0.0F);
}

TEST(IoUTest, HalfOverlapKnownValue) {
    // aは(0,0)-(10,10)、bは(5,0)-(15,10)。重なりは5x10=50、和集合は150。IoU=1/3。
    const cv::Rect2f a(0, 0, 10, 10);
    const cv::Rect2f b(5, 0, 10, 10);
    EXPECT_NEAR(IoU(a, b), 1.0F / 3.0F, 1e-5F);
}

TEST(IoUTest, ZeroAreaBoxesDoNotDivideByZero) {
    const cv::Rect2f a(0, 0, 0, 0);
    const cv::Rect2f b(0, 0, 0, 0);
    EXPECT_FLOAT_EQ(IoU(a, b), 0.0F);
}

TEST(NonMaxSuppression, EmptyInputReturnsEmpty) {
    EXPECT_TRUE(NonMaxSuppression({}, 0.45F).empty());
}

TEST(NonMaxSuppression, SuppressesOverlappingLowerScoreSameClass) {
    Detection high;
    high.box = cv::Rect2f(0, 0, 10, 10);
    high.class_id = 0;
    high.score = 0.9F;

    Detection low;
    low.box = cv::Rect2f(1, 1, 10, 10);  // 高スコアと大きく重なる
    low.class_id = 0;
    low.score = 0.5F;

    const auto kept = NonMaxSuppression({high, low}, 0.45F);
    ASSERT_EQ(kept.size(), 1U);
    EXPECT_FLOAT_EQ(kept[0].score, 0.9F);
}

TEST(NonMaxSuppression, DifferentClassesBothSurvive) {
    Detection a;
    a.box = cv::Rect2f(0, 0, 10, 10);
    a.class_id = 0;
    a.score = 0.9F;

    Detection b;
    b.box = cv::Rect2f(0, 0, 10, 10);  // 完全に重なるがクラスが違う
    b.class_id = 1;
    b.score = 0.8F;

    const auto kept = NonMaxSuppression({a, b}, 0.45F);
    EXPECT_EQ(kept.size(), 2U);
}

TEST(NonMaxSuppression, BoxesBelowThresholdBothSurvive) {
    Detection a;
    a.box = cv::Rect2f(0, 0, 10, 10);
    a.class_id = 0;
    a.score = 0.9F;

    Detection b;
    b.box = cv::Rect2f(9, 0, 10, 10);  // IoU = 1/19 ≈ 0.053、閾値未満
    b.class_id = 0;
    b.score = 0.8F;

    const auto kept = NonMaxSuppression({a, b}, 0.3F);
    EXPECT_EQ(kept.size(), 2U);
}
