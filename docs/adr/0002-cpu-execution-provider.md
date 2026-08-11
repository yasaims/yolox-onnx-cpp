# ADR 0002: CPU Execution Provider に限定する

## Context

ONNX RuntimeはCUDA/TensorRT/DirectML等の複数Execution Provider (EP)を選択できる。GPU EPは推論を高速化できる一方、CUDAツールキットやドライバのバージョン整合など環境構築のハードルが高く、他者が「READMEどおりに動かす」体験を損ないやすい。本プロジェクトの目的はポートフォリオとしての再現性・実証性であり、実行速度そのものの最適化は主目的ではない。

## Decision

CPU Execution Providerのみをサポートする。ONNX RuntimeのCPU向けビルド（プリビルト or MSYS2パッケージ）を使用し、GPU EP（CUDA/TensorRT/DirectML等）は実装しない。

## Consequences

- 環境構築がOS標準のCPUのみで完結し、READMEの手順が「コピペで通る」ことを維持しやすい
- 大きな画像・高フレームレート動画では推論速度に制約が出る。動画対応（Phase 4）ではFPS計測により実測値を正直に提示する
- GPU対応の余地はREADMEに明記し、将来的な拡張ポイントとして残す
