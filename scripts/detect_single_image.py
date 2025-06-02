# scripts/detect_single_image.py

"""
detect_single_image.py
======================
单张图片目标检测推理脚本。
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
logging.basicConfig(filename='./outputs/detect_single.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 设置参数
    model_version = 'yolov11'  # 可改为 'yolov8' / 'yolov12' / 'rtdetr'
    image_path = './test_images/911004002069_20240601.jpg'

    detector = Detector(model_version=model_version, device='cuda')
    
    start_time = time.time()
    boxes, classes, scores = detector.predict(image_path, conf_thres=0.5)
    end_time = time.time()

    logging.info(f"推理耗时: {end_time - start_time:.2f}秒")
    for i, (cls, score) in enumerate(zip(classes, scores)):
        logging.info(f"检测到目标 {i}: 类别={cls.item():.2f}, 置信度={score.item():.2f}")

    print(f"检测完成，总共检测到 {len(classes)} 个目标。")

if __name__ == '__main__':
    main()
