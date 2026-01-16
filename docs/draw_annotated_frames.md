# draw_annotated_frames.py 使い方

`draw_annotated_frames.py` は、`_bbox_video_widget.py`（動画アノテーションウィジェット）で作成した YOLO 形式のアノテーションをもとに、元動画のフレームを再読み込みして、クラスごとの色でバウンディングボックスを書き込んだ画像を保存するスクリプトです。

- フレーム画像をクロップ保存していても、元動画から該当フレームを再取得するため、元サイズの描画画像を生成できます。
- クラスごとに安定した色で描画され、`class.yaml` があればラベル（ID:クラス名）も描画します。

---

## インストール/前提

- Python 3.8+
- 依存パッケージ
  - 必須: `opencv-python`（`cv2`。矩形描画・画像保存で使用）
  - 任意: `napari-video`（`VideoReaderNP`。動画のランダムアクセス高速化に使用）

```
pip install opencv-python
# （任意）高速化したい場合
pip install napari-video
```

---

## 基本的な考え方

- アノテーションは YOLO 形式（`class cx cy w h`、正規化座標）を想定します。
- アノテーションファイル名は `img*.txt` 固定ではありません。ファイル名中に含まれる数字列のうち「最長の数字列（同長は後方優先）」をフレーム番号として扱います。
  - 例: `frame_000123.txt` → フレーム番号 123（ゼロ埋め長は 6）
  - 例: `cam1-12-000045.txt` → フレーム番号 45（ゼロ埋め長は 6）
- クラス名は `ann_dir/class.yaml` の `names` 辞書を参照します（例: `{ names: { 0: cat, 1: dog } }`）。

---

## 使い方

### 単一動画を処理（注釈フォルダは自動）

```
python draw_annotated_frames.py --video /path/to/video.mp4
```

- 注釈フォルダは既定で `<videoの拡張子除去>`（例: `/path/to/video`）を参照します。
- 出力先は既定で `<ann_dir>/<video名>_annotated/`（例: `/path/to/video/video_annotated/`）。

### 単一動画を処理（注釈フォルダを明示）

```
python draw_annotated_frames.py --video /path/to/video.mp4 --ann_dir /path/to/video
```

- 注釈フォルダを明示したい場合に使用します。
- 出力先の既定は同じく `<ann_dir>/<video名>_annotated/` です。

### ディレクトリ内の全動画を一括処理

```
python draw_annotated_frames.py --video /path/to/dir
```

- `--video` にディレクトリを渡すと、そのディレクトリ直下の動画（`*.mp4, *.avi, *.mov, *.mkv, *.wmv, *.flv, *.webm`）をすべて処理します。
- 各動画に対して注釈フォルダは `<videoの拡張子除去>` を探し、存在する場合のみ描画します。
- 出力先は既定で `<ann_dir>/<video名>_annotated/`。`--out_dir` を与えた場合は `<out_dir>/<video名>_annotated/` に書き込みます。

---

## 主なオプション

- `--video`（必須）: 動画ファイルパス、もしくは動画を含むディレクトリ
- `--ann_dir`（任意）: 注釈ディレクトリ（単一動画モードのみ有効。指定がなければ `<videoの拡張子除去>`）
- `--out_dir`（任意）: 出力ディレクトリ
  - 単一動画: 指定したパスにそのまま保存
  - ディレクトリモード: `<out_dir>/<video名>_annotated/` に保存
- `--thickness`（任意、既定 2）: バウンディングボックスの線幅（px）
- `--no-label`（任意）: クラスラベル（ID:名前）の描画を無効化

例:
```
# 出力先を明示
python draw_annotated_frames.py --video /data/v.mp4 --out_dir /data/output

# クラスラベル無し、線幅3
python draw_annotated_frames.py --video /data/v.mp4 --no-label --thickness 3

# ディレクトリ一括、共通出力ルートを指定
python draw_annotated_frames.py --video /data/videos --out_dir /data/annotated_outputs
```

---

## 入出力の仕様

- 入力注釈:
  - `ann_dir/*.txt` を走査し、`class cx cy w h`（正規化）行をパース
  - ファイル名 stem 中の数字列からフレーム番号を抽出（最長、同長は後方優先）
  - `class.yaml` があればクラス名を描画に使用
- 出力画像:
  - 1 アノテーションファイルにつき 1 画像を生成
  - ファイル名は `<元注釈ファイルのstem>_annotated.png`
  - 既定の保存先は `<ann_dir>/<video名>_annotated/`

---

## 注意・補足

- `cv2` が必須です。未インストールの場合は `pip install opencv-python` を実行してください。
- `napari-video`（`VideoReaderNP`）がインストールされていればフレームアクセスが高速化されますが、未インストールでも `cv2.VideoCapture` で動作します。
- アノテーションファイルのタイムスタンプ順ではなく、抽出したフレーム番号順に処理・保存します。
- 既に同名の出力ファイルが存在する場合は上書きされます。
- クラスカラーは ID に基づいて固定パレット→HSV で循環的に割り当てます。

---

## トラブルシューティング

- 「注釈が見つからない」
  - `--video` と対応する注釈フォルダ（既定は `<videoの拡張子除去>`）に `.txt` が存在するか確認してください。
- 「動画が開けない」
  - `cv2.VideoCapture` で開ける形式か、コーデックが環境でサポートされているか確認してください。
- 「クラス名が表示されない」
  - `ann_dir/class.yaml` に `names` 辞書が存在するか確認してください（キーは数値/文字列どちらでも可）。

---

## スクリプトの場所

- ルート直下: `draw_annotated_frames.py`

---

## ライセンス

- 本リポジトリのライセンスに従います。
