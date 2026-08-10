#include "preprocess/letterbox.hpp"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

using yolox::preprocess::Letterbox;
using yolox::preprocess::LetterboxResult;
using yolox::preprocess::ToChwFloat;

TEST(Letterbox, WideImageRatioAndOutputSize) {
    // 640x480 (横長) -> 416x416。ratio は縦横どちらの制約が厳しいかで決まる。
    cv::Mat src(480, 640, CV_8UC3, cv::Scalar(10, 20, 30));
    const LetterboxResult result = Letterbox(src, cv::Size(416, 416));

    EXPECT_FLOAT_EQ(result.info.ratio, 416.0F / 640.0F);
    EXPECT_EQ(result.image.cols, 416);
    EXPECT_EQ(result.image.rows, 416);
}

TEST(Letterbox, TallImageRatio) {
    // 480x640 (縦長) -> 416x416。
    cv::Mat src(640, 480, CV_8UC3, cv::Scalar(10, 20, 30));
    const LetterboxResult result = Letterbox(src, cv::Size(416, 416));

    EXPECT_FLOAT_EQ(result.info.ratio, 416.0F / 640.0F);
    EXPECT_EQ(result.image.cols, 416);
    EXPECT_EQ(result.image.rows, 416);
}

TEST(Letterbox, SquareSameSizeNoScaling) {
    cv::Mat src(416, 416, CV_8UC3, cv::Scalar(10, 20, 30));
    const LetterboxResult result = Letterbox(src, cv::Size(416, 416));

    EXPECT_FLOAT_EQ(result.info.ratio, 1.0F);
    // パディング領域がないので全画素が元画像由来 (114で埋められない)。
    const cv::Vec3b corner = result.image.at<cv::Vec3b>(415, 415);
    EXPECT_EQ(corner, cv::Vec3b(10, 20, 30));
}

TEST(Letterbox, PaddingIsBottomRightOnly) {
    // 800x400 (横長) -> 416x416。パディングは右と下にのみ付く。
    cv::Mat src(400, 800, CV_8UC3, cv::Scalar(10, 20, 30));
    const LetterboxResult result = Letterbox(src, cv::Size(416, 416), 114);

    // 左上端は元画像由来の値。
    const cv::Vec3b top_left = result.image.at<cv::Vec3b>(0, 0);
    EXPECT_EQ(top_left, cv::Vec3b(10, 20, 30));

    // 右下端はパディング値 (114)。
    const cv::Vec3b bottom_right = result.image.at<cv::Vec3b>(415, 415);
    EXPECT_EQ(bottom_right, cv::Vec3b(114, 114, 114));
}

TEST(ToChwFloat, KeepsBgrOrderNoColorConversion) {
    // 単一画素画像。BGR = (10, 20, 30)。RGB変換していなければ
    // CHWの先頭平面(チャンネル0)は Blue=10 になるはず。
    cv::Mat src(1, 1, CV_8UC3, cv::Scalar(10, 20, 30));
    const std::vector<float> chw = ToChwFloat(src);

    ASSERT_EQ(chw.size(), 3U);
    EXPECT_FLOAT_EQ(chw[0], 10.0F);  // B
    EXPECT_FLOAT_EQ(chw[1], 20.0F);  // G
    EXPECT_FLOAT_EQ(chw[2], 30.0F);  // R
}
