import os
import cv2
import logging
import sys
from PIL import Image 
import time
import psutil
import torch
import math
import csv

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到sys.path
sys.path.append(project_root)

from utils.yolo_utils import YOLOModel
from utils.cogvlm_inference import is_half_character

# 配置日志
log_file = "0906.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),  # 日志写入文件
    logging.StreamHandler()  # 同时输出到控制台
])

def log_memory_usage():
    # GPU 内存监控
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)
        logging.info(f"当前已分配的 GPU 内存: {allocated_memory:.2f} MB")
        logging.info(f"当前已预留的 GPU 内存: {reserved_memory:.2f} MB")
    
    # CPU 内存监控
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"当前进程的内存使用 (RSS): {mem_info.rss / (1024 ** 2):.2f} MB")

def generate_possible_readings(digit):
    if digit == 0.5:
        return [0, 1]
    elif digit == 1.5:
        return [1, 2]
    elif digit == 2.5:
        return [2, 3]
    elif digit == 3.5:
        return [3, 4]
    elif digit == 4.5:
        return [4, 5]
    elif digit == 5.5:
        return [5, 6]
    elif digit == 6.5:
        return [6, 7]
    elif digit == 7.5:
        return [7, 8]
    elif digit == 8.5:
        return [8, 9]
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

    readings = []
    for c in cls:
        # 处理可能为字符串的小数
        if isinstance(c, str) and '.' in c:
            readings.append(float(c))  # 直接将带小数的字符串转换为浮点数
        else:
            readings.append(yolo_class_map[int(c)])  # 使用类映射来获取数字

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


def process_image(image_path, yolo_model, save_dir, low_conf_threshold=0.6):
    """
    整个水表读数推理流程，包括YOLOv8检测、预处理、CogVLM2推理、决策、后处理等。
    :param image_path: 图片路径
    :param yolo_model: YOLOv8模型
    :param save_dir: 低置信度图片保存的目录
    :param low_conf_threshold: 置信度阈值，低于该值的字符会被存储
    :return: 水表最终读数结果
    """
    total_start_time = time.time()  # 开始总计时
    logging.info(f"开始处理图片: {image_path}")

    # 记录初始内存使用情况
    log_memory_usage()

    # 1. 使用 YOLOv8 进行推理
    yolo_start_time = time.time()
    results = yolo_model.predict(image_path)
    yolo_end_time = time.time()

    # 记录 YOLOv8 推理后的内存使用情况
    log_memory_usage()

    if hasattr(results[0], 'boxes'):
        boxes = results[0].boxes.xyxy  # 获取检测框坐标
        confidences = results[0].boxes.conf  # 获取置信度
        classes = results[0].boxes.cls  # 获取类别

        # logging.info(f"检测到的框: {boxes}")
        # logging.info(f"置信度: {confidences}")
        # logging.info(f"类别: {classes}")
    else:
        logging.error(f"YOLOv8 推理结果不包含 'boxes' 属性，请检查推理输出结构。")
        return "推理失败"
    
    # 2. 一系列预处理操作
    preproc_start_time = time.time()
    # 检查检测框数量
    if len(boxes) < 4:
        logging.error(f"图片 {image_path} 中的检测框数量少于 4，返回异常值。")
        return "异常"

    # 按照 x 坐标排序框，确保框按水平方向排列
    boxes, confidences, classes = zip(*sorted(zip(boxes, confidences, classes), key=lambda item: item[0][0]))

    # 计算除第一个检测框外其余框的平均面积
    avg_area = sum((box[2] - box[0]) * (box[3] - box[1]) for box in boxes[1:]) / (len(boxes) - 1)

    # 如果第一个框面积小于平均面积的80%，跳过该框
    if (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1]) < 0.8 * avg_area:
        boxes = boxes[1:]
        confidences = confidences[1:]
        classes = classes[1:]
        logging.info(f"跳过第一个框，因为面积较小。")

    if len(boxes) < 4:
        logging.error(f"图片 {image_path} 中有效检测框数量不足，返回异常值。")
        return "异常"
    preproc_end_time = time.time()

    final_reading = []

    # 3. 针对每个框使用 CogVLM2 进行推理
    cogvlm_total_start_time = time.time()
    for i, box in enumerate(boxes):
        cogvlm_char_start_time = time.time()  # 开始处理单个字符框的时间
        x_min, y_min, x_max, y_max = map(int, box[:4])
        conf = confidences[i]  # YOLOv8 的置信度
        cls = int(classes[i])  # YOLOv8 类别
        digit_image = cv2.imread(image_path)[y_min:y_max, x_min:x_max]  # 裁剪出数字框

        # 将 `numpy.ndarray` 转换为 `PIL.Image`
        digit_image_pil = Image.fromarray(cv2.cvtColor(digit_image, cv2.COLOR_BGR2RGB))

        # 使用 CogVLM2 进行推理
        cogvlm_start_time = time.time()
        digit_result = is_half_character(digit_image_pil)
        cogvlm_end_time = time.time()

        logging.info(f"YOLOv8 类别: {cls}, 置信度: {conf}, CogVLM2 结果: {digit_result}")

        # 决策流程：根据 YOLOv8 置信度和 CogVLM2 的结果决定字符类别
        if conf > 0.95:
            # 置信度很高，使用 YOLOv8 的结果
            final_reading.append(cls)
        elif 0.6 < conf <= 0.95:
            # 置信度中等，结合 CogVLM2 结果
            if digit_result["is_half_character"]:
                final_reading.append(digit_result["value"])
            else:
                final_reading.append(cls)
        else:
            # 置信度低，标记异常并保存图片
            save_image(image_path, save_dir)
            logging.warning(f"图片 {image_path} 中的字符 {cls} 置信度过低 ({conf})，保存图片并标记为异常。")
            return f"异常：低置信度字符 {cls}, 置信度 {conf}"
        
        cogvlm_char_end_time = time.time()
        
        # 记录单个字符框的时间和 CogVLM2 推理时间
        logging.info(f"单个字符框决策时间: {cogvlm_char_end_time - cogvlm_char_start_time:.4f} 秒")
        logging.info(f"CogVLM2 推理时间: {cogvlm_end_time - cogvlm_start_time:.4f} 秒")
   
    cogvlm_total_end_time = time.time()

    # 4. 根据组合规则生成最终的读数
    final_reading = combine_readings(final_reading)
    
    # 记录图片的最终水表读数
    logging.info(f"图片 {image_path} 的最终水表读数结果: {final_reading}")

    total_end_time = time.time()
   
    # 记录推理结束后的内存使用情况
    log_memory_usage()

    # 记录每个步骤的耗时
    logging.info(f"YOLOv8 推理时间: {yolo_end_time - yolo_start_time:.4f} 秒")
    logging.info(f"预处理时间: {preproc_end_time - preproc_start_time:.4f} 秒")
    logging.info(f"CogVLM2 推理总时间: {cogvlm_total_end_time - cogvlm_total_start_time:.4f} 秒")
    logging.info(f"总耗时: {total_end_time - total_start_time:.4f} 秒")
    
    return final_reading

def save_image(image_path, save_dir):
    """
    保存低置信度的水表图片。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, image_name)
    cv2.imwrite(save_path, cv2.imread(image_path))
    logging.info(f"保存低置信度图片到 {save_path}")

def process_folder(folder_path, yolo_model, save_dir, csv_file_path):
    """
    处理文件夹内所有图片，并将图片的最终读数保存到CSV文件中
    :param folder_path: 图片文件夹路径
    :param yolo_model: YOLOv8模型
    :param save_dir: 低置信度图片保存的目录
    :param csv_file_path: 保存结果的CSV文件路径
    """
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ['image_name', 'results']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for image_name in os.listdir(folder_path):
            if image_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(folder_path, image_name)
                final_reading = process_image(image_path, yolo_model, save_dir)
                
                writer.writerow({'image_name': image_name, 'results': final_reading})
                logging.info(f"图片 {image_name} 的读数结果已保存到 CSV 文件。")


if __name__ == "__main__":
    # 初始化模型
    yolo_model = YOLOModel(model_version='v11')  # 使用 YOLOv8 模型

    # 文件夹路径
    folder_path = "/home/zy/0.Data/water-try/spilt/val/images"  # 替换为图片文件夹路径
    low_conf_save_dir = "/home/zy/1.Code/new_water_meter_recognition/outputs/low_confidence/YOLOv11n"  # 替换为保存低置信度图片的文件夹
    csv_file_path = "/home/zy/1.Code/new_water_meter_recognition/outputs/0906_results.csv"  # 替换为保存CSV的文件路径

    # 处理文件夹内的所有图片
    process_folder(folder_path, yolo_model, low_conf_save_dir, csv_file_path)
