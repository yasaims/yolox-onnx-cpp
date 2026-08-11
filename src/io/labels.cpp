#include "io/labels.hpp"

#include <fstream>
#include <stdexcept>

#include "io/coco_labels.hpp"

namespace yolox::io {

namespace {

std::string Trim(const std::string& s) {
    const size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

}  // namespace

std::vector<std::string> DefaultCocoLabels() {
    return std::vector<std::string>(kCocoClassNames.begin(), kCocoClassNames.end());
}

std::vector<std::string> LoadLabels(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open labels file: " + path);
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line)) {
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        labels.push_back(trimmed);
    }

    if (labels.empty()) {
        throw std::runtime_error("Labels file contains no entries: " + path);
    }
    return labels;
}

std::string LabelFor(const std::vector<std::string>& labels, int class_id) {
    if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
        return labels[static_cast<size_t>(class_id)];
    }
    return "class_" + std::to_string(class_id);
}

}  // namespace yolox::io
