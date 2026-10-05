# ADR 0004: NMS (Non-Maximum Suppression) を自前実装する

## Status

Accepted

## Context

NMS は OpenCV の `cv::dnn::NMSBoxes` を使えば数行で済む。しかし本プロジェクトは「C++ でアルゴリズムを実装し、テストで担保できる」ことの実証を目的としており、IoU 計算と貪欲法による抑制という検出パイプラインの核心を既製関数に任せると、その価値もテスト対象も失われる。

- **`cv::dnn::NMSBoxes`**: 上記の理由に加え、`opencv_dnn` への依存が増えるため不採用

## Decision

`src/postprocess/nms.cpp` に **クラス別 greedy NMS** を自前実装する。スコア降順に走査し、同一クラス内で IoU が閾値を超える後続候補を抑制する。フィルタ後の候補は数十件程度なので、**素朴な O(n²) のまま可読性を優先する**。`IoU` は独立した関数とし、`tests/test_nms.cpp` で境界条件を含めて検証する。

## Consequences

- `opencv_dnn` をリンクせずに済む
- クラス別抑制やスコア融合などの拡張を自分のコードで制御できる
- PyTorch 版 YOLOX の NMS とは直接比較していない。挙動の裏付けは、numpy による独立再実装との突き合わせ (ADR [0005](0005-test-strategy.md)) で行う
