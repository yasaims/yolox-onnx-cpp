#pragma once

#include <string>
#include <vector>

namespace yolox::io {

// COCO 80 既定ラベル。
std::vector<std::string> DefaultCocoLabels();

// 1行1ラベル。前後空白をトリムし、空行と # 始まりは無視。
// 開けない / 有効ラベルが無い場合は std::runtime_error。
std::vector<std::string> LoadLabels(const std::string& path);

// class_id のラベル名。範囲外は "class_<id>"。
std::string LabelFor(const std::vector<std::string>& labels, int class_id);

}  // namespace yolox::io
