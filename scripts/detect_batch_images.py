# scripts/detect_batch_images.py

"""
detect_batch_images.py
======================
批量目标检测推理脚本。
支持YOLOv8/v11/v12和RT-DETR模型。
"""

import sys
import os

# 添加项目根路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import logging
import time
from utils.detection_model_utils import Detector

# 日志配置
os.makedirs('./outputs', exist_ok=True)
logging.basicConfig(filename='./outputs/detect_batch.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def is_image_file(filename):
    """判断文件是否为图片格式"""
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)

def main():
    # 设置参数
    model_version = 'yolov11'  # 可改为 'yolov8' / 'yolov12' / 'rtdetr'
    image_folder = './test_images'  # 图片文件夹路径

    detector = Detector(model_version=model_version, device='cuda')

    total_images = 0
    total_detections = 0

    for filename in os.listdir(image_folder):
        if not is_image_file(filename):
            continue

        image_path = os.path.join(image_folder, filename)
        logging.info(f"处理图片: {filename}")

        try:
            start_time = time.time()
            boxes, classes, scores = detector.predict(image_path, conf_thres=0.5)
            end_time = time.time()

            logging.info(f"推理耗时: {end_time - start_time:.2f}秒")
            for i, (cls, score) in enumerate(zip(classes, scores)):
                logging.info(f"检测到目标 {i}: 类别={cls.item():.2f}, 置信度={score.item():.2f}")

            total_images += 1
            total_detections += len(classes)

        except Exception as e:
            logging.error(f"处理图片 {filename} 时出错: {e}")

    logging.info(f"批量推理完成，总共处理图片数: {total_images}，总检测目标数: {total_detections}")
    print(f"批量推理完成，总共处理图片数: {total_images}，总检测目标数: {total_detections}")

if __name__ == '__main__':
    main()
