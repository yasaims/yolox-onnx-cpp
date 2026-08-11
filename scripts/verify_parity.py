#!/usr/bin/env python3
"""Verify numeric parity between the C++ CLI and an independent Python reference.

This does NOT call into the C++ implementation. It re-implements letterbox
preprocessing, grid/stride decoding, and NMS from scratch in numpy, following
the same spec documented in CLAUDE.md ("既知のハマりどころ"):

  - preprocessing stays BGR, no /255 normalization
  - letterbox padding is bottom-right only (no offset to subtract back)
  - grid order is stride-ascending (8, 16, 32), y-outer / x-inner within a stride

The C++ side is exercised by running the built `yolox_onnx_cpp` CLI and
parsing its stdout detection lines - the binary itself is not modified.
Because the two implementations are independent, agreement on the final
detections is evidence the three specs above are actually consistent between
languages, not just internally self-consistent.

Usage:
    python -m pip install -r scripts/requirements-dev.txt
    python scripts/verify_parity.py [--model nano|tiny|path/to/model.onnx]
        [--input tests/data/test.jpg] [--size 416] [--score-thr 0.30]
        [--nms-thr 0.45] [--bin path/to/exe] [--atol 0.05] [--verbose]

--size は省略するとモデルの入力shapeから決まる。YOLOX 0.1.1rc0 の配布ONNX
(nano/tiny とも) は入力が [1, 3, 416, 416] に固定されているため、--size で
他の値を指定しても推論できない。他サイズを試す場合は動的軸で再エクスポートした
モデルを --model にパスで渡す。
"""
from __future__ import annotations

import argparse
import glob
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_model import MODELS, download  # noqa: E402

STRIDES = (8, 16, 32)

DETECTION_LINE_RE = re.compile(
    r"class=(?P<class_id>-?\d+)\s+score=(?P<score>[\d.eE+-]+)\s+"
    r"x=(?P<x>[\d.eE+-]+)\s+y=(?P<y>[\d.eE+-]+)\s+"
    r"w=(?P<w>[\d.eE+-]+)\s+h=(?P<h>[\d.eE+-]+)"
)


# --- Python参照実装 (letterbox.cpp / decode.cpp / nms.cpp を独立に再実装) --------

def letterbox_ref(image: np.ndarray, target_size: int, pad_value: int = 114):
    """letterbox.cpp と同じ手順: アスペクト比維持リサイズ + 右下パディングのみ。

    BGRのまま・色変換なし・255スケールのまま(正規化なし)がYOLOXの前処理仕様。
    """
    h, w = image.shape[:2]
    ratio = min(target_size / h, target_size / w)
    resized_w = int(w * ratio)
    resized_h = int(h * ratio)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    padded[:resized_h, :resized_w] = resized

    chw = padded.astype(np.float32).transpose(2, 0, 1)  # BGR, 0-255スケール
    return chw, ratio


def generate_grid_strides_ref(input_size: int, strides=STRIDES):
    """stride昇順・各stride内はy外側/x内側の順でアンカー座標を生成する。

    input_size が stride で割り切れない場合、`input_size // stride` が切り捨てられて
    モデル側のグリッド数と静かにズレる。--size 側で弾いているが、この関数を直接
    呼ばれた場合に備えてここでも検査する。
    """
    for stride in strides:
        if input_size % stride != 0:
            raise ValueError(
                f"input_size={input_size} is not divisible by stride {stride} "
                f"(strides={tuple(strides)})"
            )

    grid_x_list, grid_y_list, stride_list = [], [], []
    for stride in strides:
        grid_w = input_size // stride
        grid_h = input_size // stride
        gy, gx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
        grid_x_list.append(gx.reshape(-1))
        grid_y_list.append(gy.reshape(-1))
        stride_list.append(np.full(grid_w * grid_h, stride))
    return (
        np.concatenate(grid_x_list),
        np.concatenate(grid_y_list),
        np.concatenate(stride_list),
    )


def decode_ref(output: np.ndarray, input_size: int, score_threshold: float):
    """output: [num_anchors, num_attrs] (5 + num_classes)。objectness/クラス確率は
    sigmoid適用済みの前提でgrid/strideデコードのみ行う。letterbox座標系のまま返す。
    """
    num_anchors, num_attrs = output.shape
    grid_x, grid_y, stride = generate_grid_strides_ref(input_size)
    if len(grid_x) != num_anchors:
        raise ValueError(
            f"anchor count mismatch: model={num_anchors}, generated grid={len(grid_x)}"
        )

    cx = (output[:, 0] + grid_x) * stride
    cy = (output[:, 1] + grid_y) * stride
    w = np.exp(output[:, 2]) * stride
    h = np.exp(output[:, 3]) * stride
    objectness = output[:, 4]
    class_scores = output[:, 5:]
    class_id = np.argmax(class_scores, axis=1)
    best_class_score = np.max(class_scores, axis=1)
    score = objectness * best_class_score

    keep = score >= score_threshold  # C++は `score < threshold` で捨てる -> 境界含む
    boxes = np.stack([cx - w / 2.0, cy - h / 2.0, w, h], axis=1)
    return boxes[keep], score[keep], class_id[keep]


def iou_ref(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def nms_ref(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float):
    """クラス別greedy NMS。nms.cppと同じくスコア降順に走査しIoU>閾値を抑制する。"""
    order = np.argsort(-scores)
    suppressed = np.zeros(len(order), dtype=bool)
    kept_indices = []

    for oi in range(len(order)):
        i = order[oi]
        if suppressed[i]:
            continue
        kept_indices.append(i)
        for oj in range(oi + 1, len(order)):
            j = order[oj]
            if suppressed[j] or class_ids[i] != class_ids[j]:
                continue
            if iou_ref(boxes[i], boxes[j]) > iou_threshold:
                suppressed[j] = True

    return kept_indices


def to_original_scale_ref(box: np.ndarray, ratio: float, original_size: tuple[int, int]):
    """letterbox座標系 -> 元画像座標系。パディングは右下のみなのでオフセット不要。"""
    x, y, w, h = box / ratio
    orig_w, orig_h = original_size
    x1 = max(0.0, min(x, orig_w))
    y1 = max(0.0, min(y, orig_h))
    x2 = max(0.0, min(x + w, orig_w))
    y2 = max(0.0, min(y + h, orig_h))
    return np.array([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)])


def run_python_reference(session: "ort.InferenceSession", image_path: Path, size: int,
                          score_thr: float, nms_thr: float, verbose: bool):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"failed to read input image: {image_path}")

    chw, ratio = letterbox_ref(image, size)
    input_tensor = chw[np.newaxis, :, :, :]  # [1, 3, size, size]

    input_name = session.get_inputs()[0].name
    (raw_output,) = session.run(None, {input_name: input_tensor})
    output = raw_output[0]  # [num_anchors, num_attrs]

    boxes, scores, class_ids = decode_ref(output, size, score_thr)
    kept = nms_ref(boxes, scores, class_ids, nms_thr)

    h, w = image.shape[:2]
    detections = []
    for i in kept:
        box = to_original_scale_ref(boxes[i], ratio, (w, h))
        detections.append({"class_id": int(class_ids[i]), "score": float(scores[i]), "box": box})

    if verbose:
        print(f"[python] {len(detections)} detections")
        for det in detections:
            x, y, bw, bh = det["box"]
            print(f"  class={det['class_id']} score={det['score']:.6f} x={x:.4f} y={y:.4f} "
                  f"w={bw:.4f} h={bh:.4f}")

    return detections


# --- モデル・入力サイズの解決 -------------------------------------------------

def resolve_model(spec: str) -> Path:
    """--model は登録済みキー (nano/tiny) と .onnx のパスの両方を受け付ける。

    パスを許すのは、動的軸で再エクスポートしたモデルなど download_model.py の
    管理外のモデルでもparityを取れるようにするため。
    """
    if spec in MODELS:
        return download(spec)

    path = Path(spec)
    if path.is_file():
        return path

    raise SystemExit(
        f"unknown model {spec!r}: 登録済みキー ({'/'.join(sorted(MODELS))}) か、"
        f"既存の .onnx ファイルパスを指定してください"
    )


def static_input_size(session: "ort.InferenceSession", model_path: Path) -> int | None:
    """モデル入力の一辺のピクセル数。動的軸なら None を返す。

    入力shapeは [N, C, H, W] 前提。動的軸の次元は int ではなく次元名の文字列
    (または None) として返ってくるので、それで静的/動的を判別する。
    """
    shape = session.get_inputs()[0].shape
    if len(shape) != 4:
        raise SystemExit(
            f"{model_path.name}: 入力shapeが {shape} で [N, C, H, W] ではありません"
        )

    height, width = shape[2], shape[3]
    if not isinstance(height, int) or not isinstance(width, int):
        return None
    if height != width:
        raise SystemExit(
            f"{model_path.name}: 入力が {height}x{width} で正方形ではありません。"
            f"この参照実装は正方形入力 (letterbox後) のみを前提にしています"
        )
    return height


def resolve_size(requested: int | None, model_size: int | None, model_path: Path) -> int:
    """--size とモデル側の入力サイズを突き合わせて実際に使うサイズを決める。

    配布ONNXは入力shapeが固定なので、--size を食い違わせるとONNX Runtimeが
    InvalidArgumentを投げる。それをそのまま素通しせず、原因と回避策を示す。
    """
    if model_size is None:
        if requested is None:
            raise SystemExit(
                f"{model_path.name} は入力サイズが動的です。--size を明示してください"
            )
        size = requested
    elif requested is None:
        size = model_size
        print(f"[info] --size 未指定のため、モデルの入力サイズ {size} を使用します")
    elif requested != model_size:
        raise SystemExit(
            f"--size {requested} は {model_path.name} の入力サイズ {model_size} と一致しません。\n"
            f"  この ONNX は入力shapeが [1, 3, {model_size}, {model_size}] に固定されているため、"
            f"他のサイズでは推論できません。\n"
            f"  他サイズを検証するには、入力H/Wを動的軸にして再エクスポートしたモデルを "
            f"--model <path> で渡してください。"
        )
    else:
        size = requested

    for stride in STRIDES:
        if size % stride != 0:
            raise SystemExit(
                f"--size {size} は stride {stride} で割り切れません (strides={STRIDES})。"
                f"grid数がモデル出力のアンカー数と一致しなくなります"
            )
    return size


# --- C++ CLIの標準出力パース ------------------------------------------------

def find_cli_binary() -> Path:
    candidates = sorted(
        glob.glob(str(REPO_ROOT / "build" / "*" / "src" / "yolox_onnx_cpp")) +
        glob.glob(str(REPO_ROOT / "build" / "*" / "src" / "yolox_onnx_cpp.exe"))
    )
    if not candidates:
        raise SystemExit(
            "yolox_onnx_cpp binary not found under build/*/src/. "
            "Build the project first, or pass --bin explicitly."
        )
    return Path(candidates[0])


def run_cpp_cli(binary: Path, model_path: Path, image_path: Path, size: int,
                 score_thr: float, nms_thr: float, verbose: bool):
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "parity_output.jpg"
        cmd = [
            str(binary), "--model", str(model_path), "--input", str(image_path),
            "--output", str(output_path), "--size", str(size),
            "--score-thr", str(score_thr), "--nms-thr", str(nms_thr),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"C++ CLI exited with code {result.returncode}:\n{result.stdout}\n{result.stderr}"
            )

    if verbose:
        print(f"[cpp] stdout:\n{result.stdout}")

    detections = []
    for line in result.stdout.splitlines():
        match = DETECTION_LINE_RE.search(line)
        if not match:
            continue
        detections.append({
            "class_id": int(match["class_id"]),
            "score": float(match["score"]),
            "box": np.array([float(match["x"]), float(match["y"]),
                              float(match["w"]), float(match["h"])]),
        })
    return detections


# --- 比較 -----------------------------------------------------------------

def compare(py_dets, cpp_dets, atol: float, score_atol: float = 1e-4) -> bool:
    if len(py_dets) != len(cpp_dets):
        print(f"[mismatch] detection count differs: python={len(py_dets)} cpp={len(cpp_dets)}")
        return False

    def sort_key(det):
        return (det["class_id"], -det["score"])

    py_sorted = sorted(py_dets, key=sort_key)
    cpp_sorted = sorted(cpp_dets, key=sort_key)

    ok = True
    max_box_err = 0.0
    max_score_err = 0.0
    for i, (py_det, cpp_det) in enumerate(zip(py_sorted, cpp_sorted)):
        if py_det["class_id"] != cpp_det["class_id"]:
            print(f"[mismatch] detection {i}: class_id python={py_det['class_id']} "
                  f"cpp={cpp_det['class_id']}")
            ok = False
            continue

        box_err = float(np.max(np.abs(py_det["box"] - cpp_det["box"])))
        score_err = abs(py_det["score"] - cpp_det["score"])
        max_box_err = max(max_box_err, box_err)
        max_score_err = max(max_score_err, score_err)

        if box_err > atol or score_err > score_atol:
            print(f"[mismatch] detection {i} (class={py_det['class_id']}): "
                  f"box_err={box_err:.4f} (atol={atol}) score_err={score_err:.6f} "
                  f"python_box={py_det['box']} cpp_box={cpp_det['box']}")
            ok = False

    if ok:
        print(f"parity OK: 検出数 {len(py_dets)} 一致、最大座標誤差 {max_box_err:.4f}px、"
              f"最大スコア誤差 {max_score_err:.6f}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="nano",
                         help=f"登録済みキー ({'/'.join(sorted(MODELS.keys()))}) "
                              f"または .onnx へのパス (default: nano)")
    parser.add_argument("--input", default=str(REPO_ROOT / "tests" / "data" / "test.jpg"))
    parser.add_argument("--size", type=int, default=None,
                         help="letterbox後の一辺のピクセル数。省略時はモデルの入力shapeから決まる")
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--nms-thr", type=float, default=0.45)
    parser.add_argument("--bin", default=None, help="Path to the yolox_onnx_cpp binary")
    parser.add_argument("--atol", type=float, default=0.05,
                         help="Max allowed per-coordinate pixel difference "
                              "(std::cout default precision rounds to ~6 significant digits, "
                              "so this is not 0)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_path = resolve_model(args.model)
    image_path = Path(args.input)
    binary = Path(args.bin) if args.bin else find_cli_binary()

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    size = resolve_size(args.size, static_input_size(session, model_path), model_path)

    print(f"[info] model={model_path} size={size} image={image_path} binary={binary}")

    py_dets = run_python_reference(session, image_path, size, args.score_thr,
                                    args.nms_thr, args.verbose)
    cpp_dets = run_cpp_cli(binary, model_path, image_path, size, args.score_thr,
                            args.nms_thr, args.verbose)

    ok = compare(py_dets, cpp_dets, atol=args.atol)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
