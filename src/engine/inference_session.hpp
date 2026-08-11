#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "engine/tensor_info.hpp"

namespace yolox::engine {

// Thin RAII wrapper around Ort::Env / Ort::Session. Deliberately kept free
// of any pre/post-processing knowledge: it accepts a flat CHW float tensor
// and returns the model's raw output tensors, nothing more.
class InferenceSession {
public:
    struct Options {
        int intra_op_num_threads = 0;  // 0 = let ONNX Runtime pick
        GraphOptimizationLevel opt_level = ORT_ENABLE_ALL;
    };

    explicit InferenceSession(const std::string& model_path, const Options& opts);
    explicit InferenceSession(const std::string& model_path) : InferenceSession(model_path, Options{}) {}

    const std::vector<TensorInfo>& inputs() const noexcept { return inputs_; }
    const std::vector<TensorInfo>& outputs() const noexcept { return outputs_; }

    // Runs inference on a single input tensor. `input_data` must stay alive
    // for the duration of this call (ONNX Runtime wraps it without copying).
    // Assumes a single-input model, which covers YOLOX; multi-input models
    // are out of scope for this wrapper.
    std::vector<Ort::Value> run(const std::vector<float>& input_data,
                                 const std::vector<int64_t>& input_shape);

private:
    Ort::Session session_;
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;

    // Kept alive for the lifetime of the session because run() hands raw
    // const char* pointers into these strings to the ORT C++ API.
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

}  // namespace yolox::engine
