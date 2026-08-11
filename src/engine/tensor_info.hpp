#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace yolox::engine {

// Metadata describing one model input or output. Dynamic dimensions (e.g.
// batch size) are kept as -1, mirroring ONNX Runtime's own convention.
struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    ONNXTensorElementDataType type;
};

}  // namespace yolox::engine
