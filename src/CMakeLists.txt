add_library(yolox_engine
  engine/inference_session.cpp
)
target_include_directories(yolox_engine PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(yolox_engine PUBLIC
  onnxruntime::onnxruntime
  PRIVATE yolox_warnings
)

add_library(yolox_preprocess
  preprocess/letterbox.cpp
)
target_include_directories(yolox_preprocess PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(yolox_preprocess PUBLIC
  opencv_core
  opencv_imgproc
  PRIVATE yolox_warnings
)

add_library(yolox_postprocess
  postprocess/decode.cpp
  postprocess/nms.cpp
)
target_include_directories(yolox_postprocess PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(yolox_postprocess PUBLIC
  opencv_core
  PRIVATE yolox_warnings
)

add_library(yolox_io
  io/draw.cpp
  io/labels.cpp
  io/result_writer.cpp
  io/cli_args.cpp
)
target_include_directories(yolox_io PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(yolox_io PUBLIC
  opencv_core
  opencv_imgproc
  yolox_postprocess
  PRIVATE yolox_warnings
)

add_library(yolox_pipeline
  pipeline/detector.cpp
)
target_include_directories(yolox_pipeline PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(yolox_pipeline PUBLIC
  yolox_engine
  yolox_preprocess
  yolox_postprocess
  PRIVATE yolox_warnings
)

add_executable(yolox_onnx_cpp
  main.cpp
)
target_link_libraries(yolox_onnx_cpp PRIVATE
  yolox_engine
  yolox_preprocess
  yolox_postprocess
  yolox_io
  yolox_pipeline
  opencv_imgcodecs
  opencv_videoio
  yolox_warnings
)

# On Windows, onnxruntime.dll must sit next to the executable at runtime.
# $<TARGET_RUNTIME_DLLS:...> is an empty list on Linux/macOS, which makes
# copy_if_different receive zero arguments and fail; limit to WIN32 (see
# same pattern in tests/CMakeLists.txt).
if(WIN32)
  add_custom_command(TARGET yolox_onnx_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_RUNTIME_DLLS:yolox_onnx_cpp>
      $<TARGET_FILE_DIR:yolox_onnx_cpp>
    COMMAND_EXPAND_LISTS
  )
endif()
