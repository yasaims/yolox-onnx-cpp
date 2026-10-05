# ADR 0003: 依存解決は「発見」と「取得」を分離する

## Status

Accepted

## Context

OpenCV / ONNX Runtime / GoogleTest の導入方法には、システムパッケージ、vcpkg、CMake による自動ダウンロード (FetchContent) などの選択肢がある。

- **FetchContent / vcpkg**: CMake 構成がネットワークやミラーの状態に左右され、URL・チェックサム管理で CMakeLists.txt が肥大化するため不採用

開発機は Windows (MSYS2 UCRT64)、CI は Linux (gcc/clang) であり、双方で同じ CMake 記述が通る必要がある。

## Decision

- **発見 (Find) は CMake の責務とする。** OpenCV と GoogleTest は標準の `find_package` を使う。ONNX Runtime のみ、MSYS2 版 (CMake config 同梱) と公式プリビルト (config 非同梱) を同じ `onnxruntime::onnxruntime` ターゲットに正規化する `cmake/Findonnxruntime.cmake` を用意する
- **取得 (Fetch) は CMake の外に置く。** ローカルは `pacman` で導入し、CI は公式プリビルト tarball を展開して `-DONNXRUNTIME_ROOT` で渡す。URL とバージョンのピン留めは README と `.github/workflows/ci.yml` で管理する
- モデルファイルの取得も同じ方針で、CMake とは独立した `scripts/download_model.py` に置く。Windows でも動くこと、parity 検証 (ADR [0005](0005-test-strategy.md)) が既に Python に依存していることから Python で実装する

## Consequences

- CMake の依存解決部分は `find_package` 呼び出しだけで済む
- **CMake 構成時にネットワークアクセスが発生しない。** 依存がシステムに導入済みならオフラインでも構成できる
- 初回セットアップ手順 (パッケージ導入またはプリビルト配置) を README で明示し、維持する責任が生じる
