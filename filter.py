import numpy as np
import argparse
import os
import cv2
from scipy.ndimage import generic_filter

def complete_depth_image(depth_image):
    """補完処理を行うメソッド。ゼロのピクセルを周囲の有効な値で補完します。"""
    
    def fill_zero_depth(values):
        center_value = values[len(values) // 2]
        if center_value == 0:
            non_zero_values = values[values > 0]
            if len(non_zero_values) > 0:
                return np.min(non_zero_values)  # 最も近い値で補完
            else:
                return 0  # 周囲に有効な値がなければそのまま
        return center_value
    
    # より広いカーネルサイズに設定（例: 15x15）
    depth_filled = generic_filter(depth_image, fill_zero_depth, size=15)
    return depth_filled

def load_depth_image(image_path):
    """深度画像を読み込むためのメソッド"""
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist.")
        return None
    
    # 深度画像としてPNGを読み込み
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if depth_image is None:
        print(f"Error: Unable to load the image {image_path}.")
    return depth_image

def save_depth_image(depth_image, save_path):
    """処理した深度画像を保存するメソッド"""
    cv2.imwrite(save_path, depth_image)
    print(f"Depth image saved at {save_path}")

def process_all_depth_images(input_folder):
    """指定フォルダ内の全てのJPGファイルをPNG形式に変換し、補完処理を行う"""
    # 指定フォルダ内のJPGファイルを取得
    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    for file in files:
        file_path = os.path.join(input_folder, file)
        print(f"Processing {file_path}...")
        
        # 深度画像の読み込み
        depth_image = load_depth_image(file_path)
        
        if depth_image is not None:
            # ゼロピクセルを補完
            completed_image = complete_depth_image(depth_image)
            
            # 新しいPNGファイル名を生成
            png_file_path = os.path.splitext(file_path)[0] + '.png'
            
            # 処理した深度画像をPNG形式で保存
            save_depth_image(completed_image, png_file_path)

if __name__ == '__main__':
    process_all_depth_images(os.path.join("../PipeIsoGen/data/real/images/test/depth/"))
