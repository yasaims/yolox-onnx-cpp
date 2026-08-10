cmake_minimum_required(VERSION 3.24)

project(yolox_onnx_cpp
  VERSION 0.1.0
  DESCRIPTION "C++ object detection inference CLI (ONNX Runtime + OpenCV)"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

# Common compiler warnings, collected in one interface library so every
# target (and the Phase 3 CI matrix) picks them up consistently.
add_library(yolox_warnings INTERFACE)
if(MSVC)
  target_compile_options(yolox_warnings INTERFACE /W4)
else()
  target_compile_options(yolox_warnings INTERFACE -Wall -Wextra -Wpedantic)
endif()

# CIではデフォルトで警告をエラー化する (ローカル開発は既定OFFのまま緩く保つ)。
option(YOLOX_WERROR "Treat warnings as errors" OFF)
if(YOLOX_WERROR)
  if(MSVC)
    target_compile_options(yolox_warnings INTERFACE /WX)
  else()
    target_compile_options(yolox_warnings INTERFACE -Werror)
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(onnxruntime REQUIRED)

add_subdirectory(src)

option(YOLOX_BUILD_TESTS "Build unit tests" ON)
if(YOLOX_BUILD_TESTS AND EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
  enable_testing()
  add_subdirectory(tests)
endif()
