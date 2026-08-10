## yolox-onnx-cpp

C++製 物体検出推論CLI (ONNX Runtime + OpenCV)

## 初期開発計画

開発中、プロジェクト方針を確認する場合には `.vscode/開発計画書.md`を参照する。
この計画書は更新せず、最新のコンテクストはCLAUDE.mdに記載すること。

## セットアップ (MSYS2 UCRT64)

依存パッケージ (OpenCV / ONNX Runtime / GoogleTest) を導入する:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-opencv mingw-w64-ucrt-x86_64-onnxruntime mingw-w64-ucrt-x86_64-gtest
```

いずれもCMake config (`find_package(... CONFIG)`) を同梱しているため、追加の設定は不要。

## モデル取得

```bash
python scripts/download_model.py            # models/yolox_nano.onnx を取得 (SHA256検証つき)
python scripts/download_model.py --model tiny
```

## ビルド

MSYS2 UCRT64シェル (またはPATHにucrt64/binを通した状態) で:

```bash
cmake --preset msys2-ucrt64-debug
cmake --build --preset msys2-ucrt64-debug
```

## 実行

```bash
./build/msys2-ucrt64-debug/src/yolox_onnx_cpp.exe \
  --model models/yolox_nano.onnx --input <画像パス> \
  [--output result.jpg] [--size 416] [--score-thr 0.30] [--nms-thr 0.45] [--verbose]
```

letterbox前処理 (アスペクト比維持 + 右下パディング) → 推論 → デコード (grid/stride適用) → NMS →
元画像スケールへの座標逆変換 → 描画、まで一気通貫で実行し、`--output` (既定 `output.jpg`) に
検出結果を描画した画像を書き出す。標準出力には検出件数と各検出のクラス・スコア・座標を表示する:

```
detections=1
  class=0 score=0.671259 x=45.9994 y=59.1137 w=355.777 h=448.329
```

`--verbose` を付けると入出力テンソルの形状と生出力の先頭10要素も表示する(Phase 1からの継続)。

## 設計判断

C++バージョン、CPU Execution Provider採用理由、依存解決の方針、NMS自前実装の意図などは
[docs/adr/](docs/adr/) を参照。
