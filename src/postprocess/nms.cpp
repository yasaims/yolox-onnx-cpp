#include "postprocess/nms.hpp"

#include <algorithm>
#include <numeric>

namespace yolox::postprocess {

float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter_area = (a & b).area();
    const float union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0.0F) {
        return 0.0F;
    }
    return inter_area / union_area;
}

std::vector<Detection> NonMaxSuppression(const std::vector<Detection>& detections,
                                          float iou_threshold) {
    std::vector<size_t> order(detections.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&detections](size_t lhs, size_t rhs) {
        return detections[lhs].score > detections[rhs].score;
    });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> kept;
    kept.reserve(detections.size());

    for (size_t oi = 0; oi < order.size(); ++oi) {
        const size_t i = order[oi];
        if (suppressed[i]) {
            continue;
        }
        kept.push_back(detections[i]);

        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            const size_t j = order[oj];
            if (suppressed[j]) {
                continue;
            }
            if (detections[i].class_id != detections[j].class_id) {
                continue;
            }
            if (IoU(detections[i].box, detections[j].box) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}

}  // namespace yolox::postprocess
