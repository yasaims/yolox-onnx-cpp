# ADR 0001: C++17 を採用する

## Context

本プロジェクトはモダンC++のスキル実証が目的。C++20は`std::span`や`concepts`など推論・画像処理コードに有用な機能を持つが、MSYS2 UCRT64のgccやチェックする範囲のツールチェーンで安定して使えるかを個別に確認する必要がある。CI (Phase 3) では複数コンパイラでのビルドを予定している。

## Decision

C++17を標準とし、`CMAKE_CXX_STANDARD 17` / `CXX_STANDARD_REQUIRED ON` / `CXX_EXTENSIONS OFF` を採用する。個別のC++20機能（例: `std::span`）は、採用する時点で対応コンパイラを確認したうえでADRを追加して記録する。

## Consequences

- `std::optional` / 構造化束縛 / `if constexpr` などC++17機能は自由に使える
- `std::span`は使わず、`std::vector<T>`への`const&`渡しで代替する
- 将来C++20機能を部分的に採用する場合は、CI環境で先にコンパイル確認してから導入する
