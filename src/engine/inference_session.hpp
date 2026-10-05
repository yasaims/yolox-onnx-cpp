#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "engine/tensor_info.hpp"

namespace yolox::engine {

// Ort::Session の薄い RAII ラッパー。前後処理は知らない。
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

    // 単一入力モデル専用。input_data は呼び出し中コピーされず参照される。
    std::vector<Ort::Value> run(const std::vector<float>& input_data,
                                 const std::vector<int64_t>& input_shape);

private:
    Ort::Session session_;
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;

    // run() が生ポインタを ORT に渡すため保持する。
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

}  // namespace yolox::engine
