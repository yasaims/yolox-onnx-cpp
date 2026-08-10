# ADR 0003: 依存解決は「発見」と「取得」を分離する

## Context

OpenCV / ONNX Runtime / GoogleTestの導入方法には複数の選択肢がある（システムパッケージマネージャ、vcpkg、CMakeによる自動ダウンロード等）。CMake内にダウンロードロジック（FetchContentによるバイナリ取得、URL・チェックサムの管理等）を持ち込むと、CMakeLists.txt自体が肥大化し、ネットワーク状況やミラーの都合でビルドが壊れやすくなる。一方で、依存の「発見」（どこにインストールされているかを見つける処理）は本質的にCMakeの責務である。

また、開発機はWindows（MSYS2 UCRT64）だが、Phase 3のCIはLinux（gcc/clang）を予定しており、双方で同じCMake記述が通る必要がある。

## Decision

- **発見（Find）**: CMakeの責務とする。OpenCVとGoogleTestは標準の`find_package`で十分カバーできる。ONNX Runtimeのみ、MSYS2版（CMake config同梱）と公式プリビルト（config非同梱、素朴なinclude/lib配置）の両方を1つの`onnxruntime::onnxruntime`ターゲットに正規化する自作モジュール`cmake/Findonnxruntime.cmake`（〜30行）を用意する
- **取得（Fetch）**: CMakeの外に置く。ローカル開発ではMSYS2の`pacman`でシステムに導入し、CIでは公式プリビルトのtarballをダウンロード・展開した上で`-DCMAKE_PREFIX_PATH`を渡す。取得ロジック・URL・バージョンピン留めはREADMEおよびCIワークフロー（Phase 3）側で管理する
- モデル取得（YOLOX ONNXファイル）も同じ思想で、CMakeとは独立した`scripts/download_model.py`に分離する

## Consequences

- `CMakeLists.txt`の依存解決部分は`find_package`呼び出し3行に収まり、可読性が高い
- ネットワークアクセスを伴う処理がCMake構成フェーズから排除され、オフライン環境でもシステム導入済みの依存があれば構成が通る
- 反面、初回セットアップの手順（パッケージ導入 or プリビルト配置）をREADMEで明示する責任がこちら側に生じる。これは「ビルド・実行手順がコピペで通ること」というREADME要件と表裏一体であり、許容する
- `scripts/download_model.py`はPython実装とした（当初計画書のシェルスクリプトから変更）。開発機がWindowsであること、およびPhase 3の`verify_parity.py`で既にPython依存が前提となることを踏まえた判断
