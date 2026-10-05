# ADR 0001: C++17 を採用する

## Status

Accepted

## Context

C++20 には `std::span` や `concepts` など推論・画像処理コードに有用な機能があるが、対象ツールチェーン (MSYS2 UCRT64 の gcc、CI の Linux gcc/clang) すべてで安定して使えるかは機能ごとに確認が要る。

## Decision

**C++17 を標準とする** (`CMAKE_CXX_STANDARD 17` / `CXX_STANDARD_REQUIRED ON` / `CXX_EXTENSIONS OFF`)。個別の C++20 機能を採用する場合は、全対象コンパイラでの対応を確認したうえで ADR を追加する。

## Consequences

- `std::optional` / 構造化束縛 / `if constexpr` などの C++17 機能は自由に使える
- `std::span` は使わず、`const std::vector<T>&` 渡しで代替する
