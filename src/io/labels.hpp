#pragma once

#include <string>
#include <vector>

namespace yolox::io {

// kCocoClassNames (coco_labels.hpp) を std::vector<std::string> に写したもの。
std::vector<std::string> DefaultCocoLabels();

// 1行1ラベルのテキストファイルを読む。前後空白はトリムし、空行と '#' 始まりの行は無視する。
// ファイルが開けない、または有効なラベルが1件も無い場合は std::runtime_error を送出する。
std::vector<std::string> LoadLabels(const std::string& path);

// class_id に対応するラベル名。範囲外は "class_<id>" にフォールバックする
// (draw.cpp が元々持っていた挙動をここに集約した)。
std::string LabelFor(const std::vector<std::string>& labels, int class_id);

}  // namespace yolox::io
