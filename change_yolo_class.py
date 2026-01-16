#!/usr/bin/env python3
"""
YOLO形式のアノテーションファイル内のクラス番号を一括変換するスクリプト
"""

import argparse
from pathlib import Path


def is_yolo_format(line):
    """
    行がYOLO形式かどうかを判定する
    YOLO形式: class_id x_center y_center width height
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return False
    
    try:
        # すべての値が数値に変換できるか確認
        class_id = int(parts[0])
        values = [float(x) for x in parts[1:]]
        
        # x_center, y_center, width, heightは0-1の範囲内であるべき
        for val in values:
            if val < 0 or val > 1:
                return False
        
        return True
    except ValueError:
        return False


def convert_class_number(file_path, new_class_number):
    """
    単一のYOLOアノテーションファイルのクラス番号を変換する
    
    Args:
        file_path: 変換するファイルのパス
        new_class_number: 新しいクラス番号
    
    Returns:
        bool: 変換が成功したかどうか
    """
    try:
        # ファイルを読み込む
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # YOLO形式でない行があるかチェック
        converted_lines = []
        is_yolo_file = False
        
        for line in lines:
            line = line.strip()
            if not line:  # 空行はスキップ
                continue
                
            if is_yolo_format(line):
                is_yolo_file = True
                parts = line.split()
                # クラス番号を新しい番号に置き換え
                parts[0] = str(new_class_number)
                converted_lines.append(' '.join(parts) + '\n')
            else:
                # YOLO形式でない行がある場合はファイルをスキップ
                return False
        
        # YOLO形式の行があった場合のみ書き込む
        if is_yolo_file and converted_lines:
            with open(file_path, 'w') as f:
                f.writelines(converted_lines)
            return True
        
        return False
        
    except Exception as e:
        print(f"エラー: {file_path} の処理中にエラーが発生しました: {e}")
        return False


def process_directory(directory_path, new_class_number):
    """
    ディレクトリ内のすべてのtxtファイルを処理する
    
    Args:
        directory_path: 処理するディレクトリのパス
        new_class_number: 新しいクラス番号
    """
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        print(f"エラー: ディレクトリ '{directory_path}' が見つかりません")
        return
    
    # 変換したファイルのカウンタ
    converted_count = 0
    skipped_count = 0
    error_count = 0
    
    # すべてのtxtファイルを再帰的に処理
    txt_files = list(dir_path.rglob("*.txt"))
    
    if not txt_files:
        print("警告: txtファイルが見つかりませんでした")
        return
    
    print(f"{len(txt_files)} 個のtxtファイルを処理します...")
    
    for txt_file in txt_files:
        result = convert_class_number(txt_file, new_class_number)
        
        if result:
            print(f"変換済み: {txt_file.relative_to(dir_path)}")
            converted_count += 1
        else:
            # ファイルの内容を確認してスキップ理由を判定
            try:
                with open(txt_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        # 空ファイル
                        skipped_count += 1
                    else:
                        # YOLO形式でないファイル
                        print(f"スキップ: {txt_file.relative_to(dir_path)} (YOLO形式でない)")
                        skipped_count += 1
            except:
                error_count += 1
    
    # 結果を表示
    print(f"\n処理完了:")
    print(f"  変換済み: {converted_count} ファイル")
    print(f"  スキップ: {skipped_count} ファイル")
    if error_count > 0:
        print(f"  エラー: {error_count} ファイル")
    print(f"  合計: {len(txt_files)} ファイル")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO形式アノテーションファイルのクラス番号を一括変換します"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="処理するディレクトリのパス"
    )
    parser.add_argument(
        "class_number",
        type=int,
        help="新しいクラス番号 (0以上の整数)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際には変更せず、処理対象のファイルを表示するだけ"
    )
    
    args = parser.parse_args()
    
    if args.class_number < 0:
        print("エラー: クラス番号は0以上の整数である必要があります")
        return
    
    if args.dry_run:
        print(f"[DRY RUN] クラス番号を {args.class_number} に変換します")
        # TODO: ドライラン機能の実装
        print("ドライラン機能は未実装です")
    else:
        process_directory(args.directory, args.class_number)


if __name__ == "__main__":
    main()