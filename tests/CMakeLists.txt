find_package(GTest REQUIRED)  # ADR 0003に従いFetchContentは使わない
include(GoogleTest)

add_executable(yolox_tests
  test_decode.cpp
  test_letterbox.cpp
  test_nms.cpp
  test_cli_args.cpp
  test_labels.cpp
  test_result_writer.cpp
)
target_link_libraries(yolox_tests PRIVATE
  yolox_preprocess
  yolox_postprocess
  yolox_io
  GTest::gtest_main
  yolox_warnings
)
gtest_discover_tests(yolox_tests)

# gtest_discover_tests はビルド時にテストバイナリを起動するため、
# 実行ファイルと同じDLLコピーが必要 (onnxruntime.dllは未リンクだがOpenCVは要る)。
# $<TARGET_RUNTIME_DLLS:...> はWindows以外では空リストになり、引数無しの
# copy_if_different がCOMMAND_EXPAND_LISTSと組み合わさって失敗しうるため、
# Windows限定にする (Linux/macOSはそもそも共有DLLコピーが不要)。
if(WIN32)
  add_custom_command(TARGET yolox_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_RUNTIME_DLLS:yolox_tests>
      $<TARGET_FILE_DIR:yolox_tests>
    COMMAND_EXPAND_LISTS
  )
endif()
