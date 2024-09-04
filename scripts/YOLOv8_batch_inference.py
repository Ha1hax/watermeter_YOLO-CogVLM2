import os
import sys
import time
import logging
import math
import csv
import numpy as np
from PIL import Image

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到sys.path
sys.path.append(project_root)

from utils.yolov8_utils import YOLOV8

# 日志配置，日志保存在outputs文件夹中
log_dir = os.path.join(project_root, 'outputs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, '0-9.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_boxes(xyxy, cls, conf, img_size):
    # 将YOLO预测的边框格式进行转换为[顶部, 左侧, 底部, 右侧]
    boxes = xyxy[:, [1, 0, 3, 2]].cpu().numpy()  # 将tensor从GPU移到CPU，然后转为numpy数组
    
    # 按照x_min (左侧)进行排序，确保顺序从左到右
    sorted_indices = np.argsort(boxes[:, 1])
    boxes = boxes[sorted_indices]
    cls = cls[sorted_indices]  # 对 cls 也进行相同的排序
    conf = conf[sorted_indices]  # 对 conf 也进行相同的排序

    return boxes, cls, conf

def generate_possible_readings(digit):
    if digit == 7.5:
        return [7, 8]
    elif digit == 9.5:
        return [9, 0]
    else:
        return [int(digit)]

def combine_readings(cls):
    yolo_class_map = {
        0: 0, 1: 0.5, 2: 1, 3: 1.5, 4: 2, 5: 2.5,
        6: 3, 7: 3.5, 8: 4, 9: 4.5, 10: 5,
        11: 5.5, 12: 6, 13: 6.5, 14: 7, 15: 7.5,
        16: 8, 17: 8.5, 18: 9, 19: 9.5
    }

    readings = [yolo_class_map[int(c)] for c in cls]

    # 判断是否只有最后一位是半字符
    if len(readings) > 1 and isinstance(readings[-1], float) and not readings[-1].is_integer() and all(isinstance(readings[i], int) for i in range(len(readings) - 1)):
        # 只有最后一位是半字符，直接向上取整（注意特殊处理9.5）
        if readings[-1] == 9.5:
            readings[-1] = 0
        else:
            readings[-1] = math.ceil(readings[-1])
        final_reading = ''.join(str(int(r)) for r in readings)
        return final_reading

    # 否则，处理连续的半字符情况
    possible_combinations = [[]]
    for r in readings:
        possible_digits = generate_possible_readings(r)
        new_combinations = []
        for combo in possible_combinations:
            for digit in possible_digits:
                new_combinations.append(combo + [digit])
        possible_combinations = new_combinations

    # 将组合转化为字符串形式的读数
    possible_readings = [''.join(map(str, combo)) for combo in possible_combinations]

    # 找出差值为1的两个读数
    valid_results = [(r1, r2) for r1 in possible_readings for r2 in possible_readings if abs(int(r1) - int(r2)) == 1]
    if valid_results:
        final_reading = max(max(valid_results, key=lambda x: max(int(x[0]), int(x[1]))))
    else:
        final_reading = ''.join(str(int(r)) for r in readings)

    return final_reading


def recognize_water_meter(img_path):
    logging.info(f"推理图片: {os.path.basename(img_path)}")
    try:
        # 读取图片
        start1_time = time.time()
        img = Image.open(img_path)
        img_size = img.size
        logging.info(f"读取图片 {os.path.basename(img_path)} 时间：{time.time() - start1_time}秒")
    except FileNotFoundError:
        logging.error(f"文件 {img_path} 不存在，请检查文件路径是否正确")
        return "file_not_found"
    except Exception as e:
        logging.error(f"读取图片 {img_path} 时发生异常: {e}")
        return "read_image_error"
    
    try:
        # 使用YOLOv8进行预测
        start2_time = time.time()
        yolo = YOLOV8(model_type='s')
        xyxy, cls, conf = yolo.predict(img)  # 获取预测的边框、类别和置信度
        logging.info(f"YOLOv8检测时间：{time.time() - start2_time}秒")

        # 记录每张图片的类别和对应的置信度
        for i, (c, conf_score) in enumerate(zip(cls, conf)):
            logging.info(f"类别: {c}, 置信度: {conf_score}")
            
    except Exception as e:
        logging.error(f"YOLOv8检测数字框时发生异常: {e}")
        return "yolov8_detection_error"

    try:
        # 预处理检测到的框
        boxes, cls, conf = preprocess_boxes(xyxy, cls, conf, img_size)
    except ValueError as e:
        logging.error(f"预处理时发生错误: {str(e)}")
        return str(e)

    # 组合读数
    final_reading = combine_readings(cls)

    logging.info(f"最终的水表读数: {final_reading}")
    return final_reading


def process_images_in_folder(folder_path, output_csv):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['image_name', 'results'])
        writer.writeheader()
        
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            result = recognize_water_meter(image_path)
            
            if result not in ["file_not_found", "read_image_error", "yolov8_detection_error"]:
                writer.writerow({'image_name': image_file, 'results': result})
            else:
                logging.error(f"Error processing image {image_file}: {result}")

if __name__ == "__main__":
    folder_path = "/home/zy/1.Code/new_water_meter_recognition/data/0-9_4-6"  # 替换为你实际的图片文件夹路径
    output_csv = os.path.join(log_dir, '0-9.csv')
    process_images_in_folder(folder_path, output_csv)
