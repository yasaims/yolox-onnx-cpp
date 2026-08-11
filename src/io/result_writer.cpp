#include "io/result_writer.hpp"

#include <cstdio>

#include "io/labels.hpp"

namespace yolox::io {

namespace {

// JSON文字列リテラル用のエスケープ。ラベル名にユーザ由来の任意文字列 (--labels) が
// 入りうるため、"、\、制御文字を確実にエスケープする。
std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

}  // namespace

void WriteText(std::ostream& os, const std::vector<postprocess::Detection>& detections,
               const std::vector<std::string>& labels, const ResultMeta& meta) {
    if (meta.frame_index >= 0) {
        os << "frame=" << meta.frame_index << " fps=" << meta.fps << "\n";
    }
    os << "detections=" << detections.size() << "\n";
    for (const auto& det : detections) {
        os << "  class=" << det.class_id << " score=" << det.score << " x=" << det.box.x
           << " y=" << det.box.y << " w=" << det.box.width << " h=" << det.box.height
           << " label=" << LabelFor(labels, det.class_id) << "\n";
    }
}

void WriteJson(std::ostream& os, const std::vector<postprocess::Detection>& detections,
               const std::vector<std::string>& labels, const ResultMeta& meta) {
    os << "{\"model\":\"" << JsonEscape(meta.model_path) << "\",\"input\":\""
       << JsonEscape(meta.input_path) << "\",\"input_size\":" << meta.input_size
       << ",\"score_threshold\":" << meta.score_threshold
       << ",\"nms_threshold\":" << meta.nms_threshold << ",\"image\":{\"width\":"
       << meta.image_size.width << ",\"height\":" << meta.image_size.height << "}";

    if (meta.frame_index >= 0) {
        os << ",\"frame\":" << meta.frame_index << ",\"fps\":" << meta.fps;
    }

    os << ",\"detections\":[";
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        os << "{\"class_id\":" << det.class_id << ",\"label\":\""
           << JsonEscape(LabelFor(labels, det.class_id)) << "\",\"score\":" << det.score
           << ",\"box\":{\"x\":" << det.box.x << ",\"y\":" << det.box.y << ",\"w\":" << det.box.width
           << ",\"h\":" << det.box.height << "}}";
        if (i + 1 < detections.size()) os << ",";
    }
    os << "]}\n";
}

}  // namespace yolox::io
