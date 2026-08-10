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
./build/msys2-ucrt64-debug/src/yolox_onnx_cpp.exe --model models/yolox_nano.onnx --input <画像パス> --verbose
```

Phase 1時点では前処理は単純リサイズのみ(letterbox・NMS・描画は未実装)。出力は生テンソルの形状・先頭要素をそのまま表示する。YOLOX-Nanoなら以下の形になる:

```
input  "images" shape=[1, 3, 416, 416]
output "output" shape=[1, 3549, 85]
output shape=[1, 3549, 85] elements=301665 first10=[...]
```

## 設計判断

C++バージョン、CPU Execution Provider採用理由、依存解決の方針などは [docs/adr/](docs/adr/) を参照。
