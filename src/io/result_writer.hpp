#pragma once

#include <ostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "postprocess/detection.hpp"

namespace yolox::io {

// 出力に添えるメタ情報 (モデル・入力・閾値・画像サイズ・動画のフレーム/FPS情報)。
struct ResultMeta {
    std::string model_path;
    std::string input_path;
    int input_size = 0;
    float score_threshold = 0.0F;
    float nms_threshold = 0.0F;
    cv::Size image_size;
    int frame_index = -1;  // 動画のとき >= 0、静止画のとき -1
    double fps = 0.0;      // 動画のとき > 0
};

// 既存の "detections=N" / "  class=.. score=.. x=.. y=.. w=.. h=.." 形式を維持し、
// 行末に "label=<name>" を追加する。scripts/verify_parity.py の正規表現が
// このトークン列・並び順に依存しているため、既存部分は変更しないこと
// (tests/test_result_writer.cpp が同じ正規表現で回帰を検知する)。
// 動画 (frame_index >= 0) のときは先頭に "frame=<i> fps=<f>" 行を出す。
void WriteText(std::ostream& os, const std::vector<postprocess::Detection>& detections,
               const std::vector<std::string>& labels, const ResultMeta& meta);

// 1フレーム分を1行のJSONとして出力する (動画は複数回呼ばれ JSON Lines になる)。
void WriteJson(std::ostream& os, const std::vector<postprocess::Detection>& detections,
               const std::vector<std::string>& labels, const ResultMeta& meta);

}  // namespace yolox::io
