import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到sys.path
sys.path.append(project_root)

from utils.yolov8_utils import YOLOV8

# 设置日志配置
log_file = os.path.join(current_dir, "yolov8_confidence_analysis.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # 日志文件保存路径
        logging.StreamHandler(sys.stdout)  # 同时输出到控制台
    ]
)

def preprocess_boxes(xyxy, cls, img_size):
    logging.info(f"预处理检测到的框：数量 = {len(xyxy)}")
    boxes = xyxy[:, [1, 0, 3, 2]].cpu().numpy()
    sorted_indices = np.argsort(boxes[:, 1])
    boxes = boxes[sorted_indices]
    cls = cls[sorted_indices]
    return boxes, cls

def recognize_water_meter(img_path):
    logging.info(f"推理图片: {os.path.basename(img_path)}")
    try:
        img = Image.open(img_path)
        img_size = img.size
        logging.info(f"图片尺寸: {img_size}")
    except FileNotFoundError:
        logging.error(f"文件 {img_path} 不存在，请检查文件路径是否正确")
        return None, None
    except Exception as e:
        logging.error(f"读取图片 {img_path} 时发生异常: {e}")
        return None, None
    
    try:
        yolo = YOLOV8(model_type='s')
        results = yolo.predict(img)
        logging.info(f"YOLOv8预测结果返回数量: {len(results)}")
        
        if len(results) == 3:
            xyxy, cls, confs = results
            logging.info(f"边界框数量: {len(xyxy)}, 类别数量: {len(cls)}, 置信度分数数量: {len(confs)}")
            confs = confs.cpu().numpy()  # 将置信度分数从 GPU 移动到 CPU 并转换为 NumPy 数组
            cls = cls.cpu().numpy()  # 将类别标签也移动到 CPU 并转换为 NumPy 数组
        else:
            logging.error(f"YOLOv8返回的结果长度不匹配: {len(results)}")
            return None, None
    except Exception as e:
        logging.error(f"YOLOv8检测数字框时发生异常: {e}")
        return None, None

    try:
        boxes, cls = preprocess_boxes(xyxy, cls, img_size)
    except ValueError as e:
        logging.error(f"预处理时发生错误: {str(e)}")
        return None, None

    return confs, cls  # 返回置信度列表和类别标签

def analyze_confidence_distribution(folder_path):
    all_confidences = []
    class_confidences = defaultdict(list)
    confidence_zones = {"high": 0, "medium": 0, "low": 0}

    # 遍历文件夹中的所有图片
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            logging.info(f"跳过非图像文件: {img_name}")
            continue
        
        logging.info(f"处理图像文件: {img_name}")
        confidences, classes = recognize_water_meter(img_path)
        if confidences is not None and classes is not None:
            logging.info(f"图像 {img_name} 的置信度分数: {confidences}")
            all_confidences.extend(confidences)

            for conf, cls in zip(confidences, classes):
                class_confidences[cls].append(conf)

                if conf >= 0.9:
                    confidence_zones["high"] += 1
                elif 0.7 <= conf < 0.9:
                    confidence_zones["medium"] += 1
                else:
                    confidence_zones["low"] += 1
        else:
            logging.warning(f"图像 {img_name} 没有置信度数据")

    # 创建保存图表的目录
    save_dir = os.path.join(current_dir, "confidence_plots")
    os.makedirs(save_dir, exist_ok=True)

    # 保存整体置信度分布图
    if all_confidences:
        logging.info(f"总置信度数据量: {len(all_confidences)}")
        plt.figure(figsize=(10, 6))
        plt.hist(all_confidences, bins=20, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Overall YOLOv8 Confidence Score Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'overall_confidence_distribution.png'))
        plt.close()  # 关闭图表以释放内存

    # 保存每个类别的置信度分布图
    for cls, confs in class_confidences.items():
        if confs:
            plt.figure(figsize=(10, 6))
            plt.hist(confs, bins=20, color='green', edgecolor='black', alpha=0.7)
            plt.title(f'YOLOv8 Confidence Score Distribution for Class {cls}')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'class_{cls}_confidence_distribution.png'))
            plt.close()  # 关闭图表以释放内存

    # 输出置信度分布区间统计
    logging.info("置信度分布区间统计:")
    logging.info(f"置信度 ≥ 0.9: {confidence_zones['high']} 次")
    logging.info(f"置信度在 0.7 - 0.9 之间: {confidence_zones['medium']} 次")
    logging.info(f"置信度 < 0.7: {confidence_zones['low']} 次")

if __name__ == "__main__":
    folder_path = "/home/zy/1.Code/new_water_meter_recognition/data/test_halfword"  # 替换为你存放254张图片的文件夹路径
    analyze_confidence_distribution(folder_path)
