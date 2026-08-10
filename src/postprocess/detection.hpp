#pragma once

#include <opencv2/core.hpp>

namespace yolox::postprocess {

struct Detection {
    cv::Rect2f box;  // x, y, width, height (OpenCV 慣習)
    int class_id = -1;
    float score = 0.0F;
};

}  // namespace yolox::postprocess
