#!/usr/bin/env python3
"""
画像とラベルファイルをtrain/valフォルダに整理するスクリプト
"""

import os
import shutil
import argparse
from pathlib import Path


def organize_files(source_folder, output_folder, split_type="train"):
    """
    指定フォルダ内（サブディレクトリ含む）の画像(.png)とラベル(.txt)ファイルを
    出力先フォルダ/images/[train|val]と出力先フォルダ/labels/[train|val]に整理する
    
    Args:
        source_folder: ソースフォルダのパス
        output_folder: 出力先フォルダのパス
        split_type: "train" または "val"
    """
    source_path = Path(source_folder)
    output_path = Path(output_folder)
    
    if not source_path.exists():
        print(f"エラー: フォルダ '{source_folder}' が見つかりません")
        return
    
    # 出力ディレクトリを作成
    images_dir = output_path / "images" / split_type
    labels_dir = output_path / "labels" / split_type
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 移動したファイルのカウンタ
    image_count = 0
    label_count = 0
    
    # ソースフォルダ内のすべてのファイルを再帰的に処理
    for file_path in source_path.rglob("*"):
        if file_path.is_file():
            # 画像ファイル(.png)の処理
            if file_path.suffix.lower() == ".png":
                dest_path = images_dir / file_path.name
                # 同名ファイルが存在する場合の処理
                if dest_path.exists():
                    base_name = file_path.stem
                    counter = 1
                    while dest_path.exists():
                        new_name = f"{base_name}_{counter}{file_path.suffix}"
                        dest_path = images_dir / new_name
                        counter += 1
                shutil.move(str(file_path), str(dest_path))
                print(f"移動: {file_path.relative_to(source_path)} -> {dest_path.relative_to(output_path)}")
                image_count += 1
            
            # ラベルファイル(.txt)の処理
            elif file_path.suffix.lower() == ".txt":
                dest_path = labels_dir / file_path.name
                # 同名ファイルが存在する場合の処理
                if dest_path.exists():
                    base_name = file_path.stem
                    counter = 1
                    while dest_path.exists():
                        new_name = f"{base_name}_{counter}{file_path.suffix}"
                        dest_path = labels_dir / new_name
                        counter += 1
                shutil.move(str(file_path), str(dest_path))
                print(f"移動: {file_path.relative_to(source_path)} -> {dest_path.relative_to(output_path)}")
                label_count += 1
    
    # 結果を表示
    print(f"\n完了:")
    print(f"  画像ファイル: {image_count} 個を {images_dir.relative_to(output_path)}/ に移動")
    print(f"  ラベルファイル: {label_count} 個を {labels_dir.relative_to(output_path)}/ に移動")


def main():
    parser = argparse.ArgumentParser(
        description="画像とラベルファイルをtrain/valフォルダに整理します"
    )
    parser.add_argument(
        "source",
        type=str,
        help="整理するソースフォルダのパス"
    )
    parser.add_argument(
        "output",
        type=str,
        help="出力先フォルダのパス"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="train",
        help="データセットの分割タイプ (デフォルト: train)"
    )
    
    args = parser.parse_args()
    
    organize_files(args.source, args.output, args.split)


if __name__ == "__main__":
    main()