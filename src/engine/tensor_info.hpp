#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace yolox::engine {

// 動的次元は -1。
struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    ONNXTensorElementDataType type;
};

}  // namespace yolox::engine
