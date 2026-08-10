## yolox-onnx-cpp

[![CI](https://github.com/yasaims/yolox-onnx-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/yasaims/yolox-onnx-cpp/actions/workflows/ci.yml)

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

## テスト実行

GoogleTestによる単体テスト (letterbox・NMS・decode) をビルドと同じプリセットで実行する:

```bash
cmake --build --preset msys2-ucrt64-debug
ctest --test-dir build/msys2-ucrt64-debug --output-on-failure
```

## Linuxでのビルド (CIと同じ手順)

```bash
sudo apt-get install -y ninja-build libopencv-dev libgtest-dev

# ONNX Runtime公式プリビルトを取得 (MSYS2のようなCONFIGパッケージが無いため)
curl -sSL -o /tmp/onnxruntime.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
mkdir -p /tmp/onnxruntime && tar -xzf /tmp/onnxruntime.tgz --strip-components=1 -C /tmp/onnxruntime

cmake --preset linux-gcc-release -DCMAKE_PREFIX_PATH=/tmp/onnxruntime
cmake --build --preset linux-gcc-release
LD_LIBRARY_PATH=/tmp/onnxruntime/lib ctest --preset linux-gcc-release
```

`linux-clang-release` プリセットも同様 (clangをインストールした上で使う)。GitHub Actions
(`.github/workflows/ci.yml`) は gcc/clang 両方でこの手順を実行する。

## 数値一致検証 (Python参照実装とのparity)

letterbox前処理・grid/strideデコード・NMSの3箇所は、Python (numpy + ONNX Runtime) で
独立に再実装した参照実装との数値一致を `scripts/verify_parity.py` で検証できる
(BGR順序・座標系・grid順序が実装間でズレていないことの裏付け)。CIには含めない
ローカル開発ツール (`docs/adr/0005-test-strategy.md` 参照)。

```bash
python -m pip install -r scripts/requirements-dev.txt
cmake --build --preset msys2-ucrt64-debug   # yolox_onnx_cpp を先にビルドしておく
python scripts/verify_parity.py --verbose
```

## 設計判断

C++バージョン、CPU Execution Provider採用理由、依存解決の方針、NMS自前実装の意図などは
[docs/adr/](docs/adr/) を参照。
