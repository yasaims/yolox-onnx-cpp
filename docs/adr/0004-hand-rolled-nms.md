# ADR 0004: NMS (Non-Maximum Suppression) を自前実装する

## Context

後処理でのNMSは、OpenCVの `cv::dnn::NMSBoxes` を使えば数行で済む。一方、本プロジェクトの目的 (`.vscode/開発計画書.md` §2) は「C++でアルゴリズムを実装しテストで担保できる」ことの実証であり、NMSはIoU計算・貪欲法による抑制という単純だが検出パイプラインの核心をなすアルゴリズムである。既製関数を呼ぶだけでは、この部分のアピール価値もテスト対象も失われる。

`cv::dnn::NMSBoxes` を使う場合、`opencv_dnn` コンポーネントへの依存が新たに増える点も考慮した (現状は `core` / `imgproc` / `imgcodecs` のみ)。

## Decision

`src/postprocess/nms.cpp` にクラス別 greedy NMS を自前実装する。スコア降順に候補を走査し、同一クラス内でIoUが閾値を超える後続候補を抑制する素朴な O(n²) アルゴリズムとし、可読性を優先する (フィルタ後の候補数は数十件程度で最適化の必要がない)。IoU計算 (`IoU`) も独立した関数として切り出し、単体テスト (`tests/test_nms.cpp`) で境界条件込みで検証する。

## Consequences

- `opencv_dnn` への依存が不要になり、リンクするOpenCVコンポーネントを `core` / `imgproc` / `imgcodecs` のみに保てる
- クラス別抑制やスコア融合など、将来の拡張 (Phase 4以降) を自分のコードで自由に制御できる
- 公式実装 (PyTorch版YOLOXのNMS) との数値的な一致は本ADRの時点では未検証。Phase 3の `verify_parity.py` で検出結果全体の一致を確認する
