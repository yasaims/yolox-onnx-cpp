# ADR 0002: CPU Execution Provider に限定する

## Status

Accepted

## Context

ONNX Runtime は CUDA / TensorRT / DirectML などの Execution Provider (EP) を選択できる。GPU EP は推論を高速化できるが、CUDA ツールキットやドライバのバージョン整合など環境構築のハードルが高く、README の手順どおりに再現できることを損ないやすい。本プロジェクトは再現性を優先し、実行速度の最適化は主目的としない。

## Decision

**CPU EP のみをサポートする。** ONNX Runtime の CPU 向けビルド (MSYS2 パッケージまたは公式プリビルト) を使い、GPU EP は実装しない。

## Consequences

- 環境構築が CPU のみで完結し、README の手順をそのまま実行すれば動く状態を保ちやすい
- 大きな画像・高フレームレート動画では推論速度が制約になる。動画処理では FPS を描画・出力して実測値を示す
- GPU 対応は現状提供していない。追加する場合は EP 選択の CLI オプションと環境別の導入手順が必要になる
