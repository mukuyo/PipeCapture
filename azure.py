import time
import numpy as np
import argparse
import datetime
import open3d as o3d
import cv2
import os
from scipy.ndimage import generic_filter

directory_name = '2021-09-09-52'



class RecorderWithCallback:

    def __init__(self, config, device, filename, align_depth_to_color):
        # Global flags
        self.flag_exit = False
        self.flag_record = False
        self.is_key_tap = False
        self.filename = filename

        self.align_depth_to_color = align_depth_to_color
        self.recorder = o3d.io.AzureKinectRecorder(config, device)
        if not self.recorder.init_sensor():
            raise RuntimeError('Failed to connect to sensor')

    def escape_callback(self, vis):
        self.flag_exit = True
        if self.recorder.is_record_created():
            print('Recording finished.')
        else:
            print('Nothing has been recorded.')
        return False

    def space_callback(self, vis):
        if self.flag_record:
            print('Recording paused. '
                  'Press [Space] to continue. '
                  'Press [ESC] to save and exit.')
            self.flag_record = False

        elif not self.recorder.is_record_created():
            if self.recorder.open_record(self.filename):
                print('Recording started. '
                      'Press [SPACE] to pause. '
                      'Press [ESC] to save and exit.')
                self.flag_record = True

        else:
            print('Recording resumed, video may be discontinuous. '
                  'Press [SPACE] to pause. '
                  'Press [ESC] to save and exit.')
            self.flag_record = True

        return False

    def complete_depth_image(self, depth_image):
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
        
        # 周囲3x3ピクセルを見てゼロピクセルを補完する
        depth_filled = generic_filter(depth_image, fill_zero_depth, size=80)
        return depth_filled

    def run(self):
        glfw_key_escape = 256
        glfw_key_space = 32
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.register_key_callback(glfw_key_space, self.space_callback)

        vis.create_window('recorder', 1920, 540)
        print("Recorder initialized. Press [SPACE] to start. "
              "Press [ESC] to save and exit.")
        
        # Create directories
        os.makedirs(f"data/{directory_name}", exist_ok=True)
        os.makedirs(f"data/{directory_name}/rgb", exist_ok=True)
        os.makedirs(f"data/{directory_name}/depth", exist_ok=True)

        count = 0
        vis_geometry_added = False
        last_save_time = time.time()
        save_interval = 0.42  # Save every 10 seconds
        
        while not self.flag_exit:
            rgbd = self.recorder.record_frame(self.flag_record,
                                              self.align_depth_to_color)
            if rgbd is None:
                continue

            # Get color image
            color_image = np.asarray(rgbd.color)
            # Convert color image to BGR for OpenCV
            color_bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # Get depth data
            depth_image = rgbd.depth
            depth_np = np.asarray(depth_image)

            # Convert depth data to uint16 format (mm units)
            depth_mm = (depth_np * 0.1).astype(np.uint16)
            center_x = depth_mm.shape[1] // 2
            center_y = depth_mm.shape[0] // 2

            # Get the depth value at the center of the image (in mm)
            center_depth_mm = depth_mm[center_y, center_x]

            # Print the center depth value
            print(f"Center depth value: {center_depth_mm} mm")
            # Check if it's time to save images
            current_time = time.time()
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                self.is_key_tap = True

            if current_time - last_save_time >= save_interval and self.is_key_tap:
                # Define cropping dimensions
                target_width = 640
                target_height = 480

                # Calculate cropping box for center of the image
                center_x, center_y = color_bgr_image.shape[1] // 2, color_bgr_image.shape[0] // 2
                crop_x1 = max(center_x - target_width // 2, 0)
                crop_y1 = max(center_y - target_height // 2, 0)
                crop_x2 = min(center_x + target_width // 2, color_bgr_image.shape[1])
                crop_y2 = min(center_y + target_height // 2, color_bgr_image.shape[0])

                # Crop color and depth images
                color_cropped = color_bgr_image[crop_y1:crop_y2, crop_x1:crop_x2]
                depth_cropped = depth_mm[crop_y1:crop_y2, crop_x1:crop_x2]

                # Save the cropped color and depth images
                color_filename = f'../PipeIsoGen/data/real/images/test/rgb/frame{count*10}.jpg'
                depth_filename = f'../PipeIsoGen/data/real/images/test/depth/frame{count*10}.png'        

                cv2.imwrite(color_filename, color_cropped)
                cv2.imwrite(depth_filename, depth_cropped)

                print(f"Saved RGB to {color_filename} and Depth to {depth_filename}")
                count += 1
                last_save_time = current_time

            cv2.imshow("Depth Image", depth_mm)
            cv2.imshow("Color Image", color_bgr_image)
            cv2.waitKey(10)

            if not vis_geometry_added:
                vis.add_geometry(rgbd)
                vis_geometry_added = True

            vis.update_geometry(rgbd)
            vis.poll_events()
            vis.update_renderer()

        self.recorder.close_record()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure kinect mkv recorder.')
    parser.add_argument('--config', type=str, help='input json kinect config')
    parser.add_argument('--output', type=str, help='output mkv filename')
    parser.add_argument('--list',
                        action='store_true',
                        help='list available azure kinect sensors')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='input kinect device id')
    parser.add_argument('-a',
                        '--align_depth_to_color',
                        action='store_true',
                        help='enable align depth image to color')
    args = parser.parse_args()

    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    if args.config is not None:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    if args.output is not None:
        filename = args.output
    else:
        filename = '{date:%Y-%m-%d-%H-%M-%S}.mkv'.format(
            date=datetime.datetime.now())
    print('Prepare writing to {}'.format(filename))

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    r = RecorderWithCallback(config, device, filename,
                             args.align_depth_to_color)
    r.run()
