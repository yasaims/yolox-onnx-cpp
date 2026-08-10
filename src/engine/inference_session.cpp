#include "engine/inference_session.hpp"

#include <memory>

#ifdef _WIN32
#include <windows.h>
#endif

namespace yolox::engine {

namespace {

// Ort::Env is process-wide; a Meyers singleton keeps exactly one alive for
// the life of the program regardless of how many InferenceSession instances
// are created.
Ort::Env& GlobalEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolox_onnx_cpp");
    return env;
}

#ifdef _WIN32
// On Windows, Ort::Session takes the model path as a wchar_t*. This applies
// under MSYS2/MinGW too, since _WIN32 is defined there as well. Uses the
// Win32 API directly rather than the deprecated std::wstring_convert.
std::wstring ToWideString(const std::string& s) {
    if (s.empty()) return std::wstring();
    const int required = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring wide(static_cast<size_t>(required), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), wide.data(), required);
    return wide;
}
#endif

Ort::Session MakeSession(const std::string& model_path, const InferenceSession::Options& opts) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(opts.intra_op_num_threads);
    session_options.SetGraphOptimizationLevel(opts.opt_level);

#ifdef _WIN32
    const std::wstring wide_path = ToWideString(model_path);
    return Ort::Session(GlobalEnv(), wide_path.c_str(), session_options);
#else
    return Ort::Session(GlobalEnv(), model_path.c_str(), session_options);
#endif
}

std::vector<int64_t> ShapeOf(const Ort::TypeInfo& type_info) {
    return type_info.GetTensorTypeAndShapeInfo().GetShape();
}

ONNXTensorElementDataType ElementTypeOf(const Ort::TypeInfo& type_info) {
    return type_info.GetTensorTypeAndShapeInfo().GetElementType();
}

}  // namespace

InferenceSession::InferenceSession(const std::string& model_path, const Options& opts)
    : session_(MakeSession(model_path, opts)) {
    Ort::AllocatorWithDefaultOptions allocator;

    const size_t num_inputs = session_.GetInputCount();
    input_names_.reserve(num_inputs);
    inputs_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_.GetInputNameAllocated(i, allocator);
        input_names_.emplace_back(name.get());
        const Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
        inputs_.push_back(TensorInfo{input_names_.back(), ShapeOf(type_info), ElementTypeOf(type_info)});
    }

    const size_t num_outputs = session_.GetOutputCount();
    output_names_.reserve(num_outputs);
    outputs_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_.GetOutputNameAllocated(i, allocator);
        output_names_.emplace_back(name.get());
        const Ort::TypeInfo type_info = session_.GetOutputTypeInfo(i);
        outputs_.push_back(TensorInfo{output_names_.back(), ShapeOf(type_info), ElementTypeOf(type_info)});
    }
}

std::vector<Ort::Value> InferenceSession::run(const std::vector<float>& input_data,
                                                const std::vector<int64_t>& input_shape) {
    const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // const_cast is safe here: ORT only reads from the input tensor. The
    // API just doesn't expose a const-correct overload.
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(input_data.data()), input_data.size(),
        input_shape.data(), input_shape.size());

    std::vector<const char*> input_name_ptrs;
    input_name_ptrs.reserve(input_names_.size());
    for (const auto& name : input_names_) {
        input_name_ptrs.push_back(name.c_str());
    }

    std::vector<const char*> output_name_ptrs;
    output_name_ptrs.reserve(output_names_.size());
    for (const auto& name : output_names_) {
        output_name_ptrs.push_back(name.c_str());
    }

    return session_.Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(), &input_tensor, 1,
                         output_name_ptrs.data(), output_name_ptrs.size());
}

}  // namespace yolox::engine
