import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs

directory_name = '2021-09-09-52'

# RealSenseパイプラインの初期化
pipeline = rs.pipeline()
config = rs.config()

# ストリームの有効化
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# パイプラインの開始
profile = pipeline.start(config)

# 深度スケールを取得（単位はメートル）
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale: {depth_scale} meters")

align_to = rs.stream.color
align = rs.align(align_to)

# フィルターの初期化
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

# フィルターパラメータの設定
decimation.set_option(rs.option.filter_magnitude, 1)
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
spatial.set_option(rs.option.holes_fill, 2)
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal.set_option(rs.option.filter_smooth_delta, 20)

def get_camera_matrix(intrinsics):
    return np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])

# カメラパラメータを取得して保存
depth_stream = profile.get_stream(rs.stream.depth)
depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

color_stream = profile.get_stream(rs.stream.color)
color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

depth_camera_matrix = get_camera_matrix(depth_intrinsics)
color_camera_matrix = get_camera_matrix(color_intrinsics)

# カメラ行列を保存
timestamp = int(time.time())
color_params_filename = f'data/cam_K.txt'
np.savetxt(color_params_filename, color_camera_matrix, fmt='%f')

count = 0
is_capture = False

# ディレクトリ作成
os.makedirs(f"data/{directory_name}/rgb", exist_ok=True)
os.makedirs(f"data/{directory_name}/depth", exist_ok=True)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # フィルターの適用
        depth_frame = decimation.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 深度スケールを適用してメートル単位に変換
        depth_image_meters = depth_image * depth_scale
        
        # カラー画像の表示
        cv2.imshow("RGB Image", (depth_image_meters * 10000).astype(np.uint16))
        
        key = cv2.waitKey(1)

        # 深度画像とカラー画像の保存
        if key & 0xFF == ord('s') or is_capture:
            current_time = time.time()
            
            if count%10 == 0:
                color_filename = f'data/{directory_name}/rgb/frame{count}.png'
                depth_filename = f'data/{directory_name}/depth/frame{count}.png'
                
                # カラー画像の保存
                cv2.imwrite(color_filename, color_image)
                
                # 深度画像の保存（mm単位）
                cv2.imwrite(depth_filename, (depth_image_meters * 100).astype(np.uint16))
                print(f"Saved frame {count}")
                
            count += 1    
            is_capture = True
        
        # 'q'キーで終了
        if key & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
