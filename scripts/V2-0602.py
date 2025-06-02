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
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.detection_model_utils import Detector
from utils.cogvlm_inference import is_half_character

log_file = "V2-0602.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])

def log_memory_usage():
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)
        logging.info(f"当前已分配的 GPU 内存: {allocated_memory:.2f} MB")
        logging.info(f"当前已预留的 GPU 内存: {reserved_memory:.2f} MB")
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"当前进程的内存使用 (RSS): {mem_info.rss / (1024 ** 2):.2f} MB")

def generate_possible_readings(digit):
    mapping = {0.5: [0,1], 1.5:[1,2], 2.5:[2,3], 3.5:[3,4], 4.5:[4,5],
               5.5:[5,6], 6.5:[6,7], 7.5:[7,8], 8.5:[8,9], 9.5:[9,0]}
    return mapping.get(digit, [int(digit)])

def combine_readings(cls):
    yolo_class_map = {0:0,1:0.5,2:1,3:1.5,4:2,5:2.5,6:3,7:3.5,8:4,9:4.5,
                      10:5,11:5.5,12:6,13:6.5,14:7,15:7.5,16:8,17:8.5,18:9,19:9.5}
    readings = [float(c) if isinstance(c,str) and '.' in c else yolo_class_map[int(c)] for c in cls]
    if len(readings)>1 and isinstance(readings[-1],float) and not readings[-1].is_integer() and all(isinstance(r,int) for r in readings[:-1]):
        readings[-1] = 0 if readings[-1]==9.5 else math.ceil(readings[-1])
        return ''.join(str(int(r)) for r in readings)
    combos = [[]]
    for r in readings:
        combos = [c+[d] for c in combos for d in generate_possible_readings(r)]
    possible = [''.join(map(str,c)) for c in combos]
    valid = [(r1,r2) for r1 in possible for r2 in possible if abs(int(r1)-int(r2))==1]
    return max(max(valid,key=lambda x:max(int(x[0]),int(x[1])))) if valid else ''.join(str(int(r)) for r in readings)

def process_image(image_path, yolo_model, save_dir, low_conf_threshold=0.6):
    total_start_time = time.time()
    logging.info(f"开始处理图片: {image_path}")
    log_memory_usage()

    yolo_start_time = time.time()
    # 调用 Detector.predict()，确保返回 (boxes, scores, classes)
    boxes, confidences, classes = yolo_model.predict(image_path)
    yolo_end_time = time.time()
    log_memory_usage()

    if len(boxes) < 4:
        logging.error(f"图片 {image_path} 中的检测框数量少于 4，返回异常值。")
        return "异常"

    zipped = list(zip(boxes, confidences, classes))
    zipped.sort(key=lambda item: item[0][0])
    boxes, confidences, classes = zip(*zipped)

    avg_area = sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes[1:]) / (len(boxes)-1)
    if (boxes[0][2]-boxes[0][0])*(boxes[0][3]-boxes[0][1]) < 0.8*avg_area:
        boxes, confidences, classes = boxes[1:], confidences[1:], classes[1:]
        logging.info(f"跳过第一个框，因为面积较小。")
    if len(boxes) < 4:
        logging.error(f"图片 {image_path} 中有效检测框数量不足，返回异常值。")
        return "异常"

    final_reading = []
    cogvlm_total_start_time = time.time()
    for i, box in enumerate(boxes):
        cogvlm_char_start_time = time.time()
        x_min,y_min,x_max,y_max = map(int, box[:4])
        conf = confidences[i]
        cls = int(classes[i])
        digit_image = cv2.imread(image_path)[y_min:y_max, x_min:x_max]
        digit_pil = Image.fromarray(cv2.cvtColor(digit_image, cv2.COLOR_BGR2RGB))
        cogvlm_start_time = time.time()
        digit_result = is_half_character(digit_pil)
        cogvlm_end_time = time.time()

        logging.info(f"YOLO 类别: {cls}, 置信度: {conf}, CogVLM2 结果: {digit_result}")
        if conf > 0.95:
            final_reading.append(cls)
        elif 0.6 < conf <= 0.95:
            final_reading.append(digit_result["value"] if digit_result["is_half_character"] else cls)
        else:
            save_image(image_path, save_dir)
            logging.warning(f"图片 {image_path} 中的字符 {cls} 置信度过低 ({conf})，保存图片并标记为异常。")
            return f"异常：低置信度字符 {cls}, 置信度 {conf}"

        cogvlm_char_end_time = time.time()
        logging.info(f"单个字符框决策时间: {cogvlm_char_end_time - cogvlm_char_start_time:.4f} 秒")
        logging.info(f"CogVLM2 推理时间: {cogvlm_end_time - cogvlm_start_time:.4f} 秒")

    cogvlm_total_end_time = time.time()
    final_reading = combine_readings(final_reading)
    logging.info(f"图片 {image_path} 的最终水表读数结果: {final_reading}")
    total_end_time = time.time()
    log_memory_usage()
    logging.info(f"YOLO 推理时间: {yolo_end_time - yolo_start_time:.4f} 秒")
    logging.info(f"CogVLM2 推理总时间: {cogvlm_total_end_time - cogvlm_total_start_time:.4f} 秒")
    logging.info(f"总耗时: {total_end_time - total_start_time:.4f} 秒")
    return final_reading

def save_image(image_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(save_dir, image_name), cv2.imread(image_path))
    logging.info(f"保存低置信度图片到 {save_dir}")

def process_folder(folder_path, yolo_model, save_dir, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['image_name', 'results'])
        writer.writeheader()
        for image_name in os.listdir(folder_path):
            if image_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(folder_path, image_name)
                final_reading = process_image(image_path, yolo_model, save_dir)
                writer.writerow({'image_name': image_name, 'results': final_reading})
                logging.info(f"图片 {image_name} 的读数结果已保存到 CSV 文件。")

if __name__ == "__main__":
    yolo_model = Detector(model_version='yolov11')
    folder_path = "/home/zy/1.Code/watermeter_YOLO-CogVLM2/datasets/images/test"
    low_conf_save_dir = "outputs/low_confidence/YOLOv11n"
    csv_file_path = "/home/zy/1.Code/watermeter_YOLO-CogVLM2/outputs/V2/0602.csv"
    process_folder(folder_path, yolo_model, low_conf_save_dir, csv_file_path)
