import os
import cv2
import logging
import sys
from PIL import Image
import time
import psutil
import torch
import csv
import pandas as pd

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.detection_model_utils import Detector
from utils.cogvlm_inference import is_half_character

log_file = "V2-0602-Updated.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])

# 加载真实标签
def load_ground_truth(csv_path):
    df = pd.read_csv(csv_path)
    gt_dict = {}
    for _, row in df.iterrows():
        gt_dict[row['image_name']] = str(row['Sequence Annotation'])
    return gt_dict

def generate_reading(annotations):
    final_reading = []
    for i, label in enumerate(annotations):
        if i == 0 and label == '9.5':
            final_reading.append(0)
        elif i == 1 and annotations[0] == '9.5' and label == '9.5':
            final_reading.append(0)
        else:
            if '.5' in label:
                final_reading.append(int(float(label)))
            else:
                final_reading.append(int(label))
    return ''.join(map(str, final_reading))

def process_image(image_path, yolo_model, save_dir, detailed_writer, ground_truth_dict, low_conf_threshold=0.6, high_conf_threshold=0.95):
    image_name = os.path.basename(image_path)
    true_seq = ground_truth_dict.get(image_name, "")

    total_start_time = time.time()
    logging.info(f"开始处理图片: {image_path}")

    yolo_start_time = time.time()
    boxes, confidences, classes = yolo_model.predict(image_path, conf_thres=0.5)
    yolo_end_time = time.time()

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

    class_mapping = {
        0: '0', 1: '0.5', 2: '1', 3: '1.5', 4: '2', 5: '2.5', 6: '3', 7: '3.5', 8: '4', 9: '4.5',
        10: '5', 11: '5.5', 12: '6', 13: '6.5', 14: '7', 15: '7.5', 16: '8', 17: '8.5', 18: '9', 19: '9.5'
    }

    annotations = []
    cogvlm_total_start_time = time.time()

    for i, box in enumerate(boxes):
        cogvlm_char_start_time = time.time()
        x_min, y_min, x_max, y_max = map(int, box[:4])
        conf = confidences[i]
        cls = int(classes[i])
        label = class_mapping[cls]

        digit_image = cv2.imread(image_path)[y_min:y_max, x_min:x_max]
        digit_pil = Image.fromarray(cv2.cvtColor(digit_image, cv2.COLOR_BGR2RGB))

        cogvlm_start_time = time.time()
        digit_result = is_half_character(digit_pil, cls)
        cogvlm_end_time = time.time()

        cogvlm_prediction = digit_result['value']

        if conf > high_conf_threshold:
            annotations.append(label)
        elif low_conf_threshold < conf <= high_conf_threshold:
            annotations.append(cogvlm_prediction if digit_result["is_half_character"] else label)
        else:
            save_image(image_path, save_dir)
            logging.warning(f"图片 {image_path} 中的字符 {label} 置信度过低 ({conf})，保存图片并标记为异常。")
            return f"异常：低置信度字符 {label}, 置信度 {conf}"

        true_char = true_seq[i] if i < len(true_seq) else ""
        yolo_correct = (label == true_char)
        cogvlm_correct = (str(cogvlm_prediction) == true_char)

        detailed_writer.writerow({
            'image_name': image_name,
            'true_label': true_char,
            'yolo_prediction': label,
            'yolo_conf': conf,
            'cogvlm_prediction': cogvlm_prediction,
            'yolo_correct': yolo_correct,
            'cogvlm_correct': cogvlm_correct
        })

        # 每个字符日志记录
        logging.info(f"图片 {image_name} 字符 {i}: YOLO类别={label}, 置信度={conf:.4f}, CogVLM2预测={cogvlm_prediction}")

        cogvlm_char_end_time = time.time()
        logging.info(f"单个字符框决策时间: {cogvlm_char_end_time - cogvlm_char_start_time:.4f} 秒")
        logging.info(f"CogVLM2 推理时间: {cogvlm_end_time - cogvlm_start_time:.4f} 秒")

    cogvlm_total_end_time = time.time()
    final_reading = generate_reading(annotations)

    # 每张图片日志记录
    logging.info(f"图片 {image_name} 的最终预测读数: {final_reading}")

    total_end_time = time.time()
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

def process_folder(folder_path, yolo_model, save_dir, csv_file_path, detailed_csv_path, gt_csv_path):
    ground_truth_dict = load_ground_truth(gt_csv_path)

    with open(csv_file_path, mode='w', newline='') as csv_file, open(detailed_csv_path, mode='w', newline='') as detailed_file:
        writer = csv.DictWriter(csv_file, fieldnames=['image_name', 'results'])
        writer.writeheader()

        detailed_writer = csv.DictWriter(detailed_file, fieldnames=['image_name', 'true_label', 'yolo_prediction', 'yolo_conf', 'cogvlm_prediction', 'yolo_correct', 'cogvlm_correct'])
        detailed_writer.writeheader()

        for image_name in os.listdir(folder_path):
            if image_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(folder_path, image_name)
                final_reading = process_image(image_path, yolo_model, save_dir, detailed_writer, ground_truth_dict)
                writer.writerow({'image_name': image_name, 'results': final_reading})
                logging.info(f"图片 {image_name} 的读数结果已保存到 CSV 文件。")

if __name__ == "__main__":
    yolo_model = Detector(model_version='yolov11')  # 确保 Detector.predict() 已启用 agnostic_nms=True
    folder_path = "/home/zy/1.Code/watermeter_YOLO-CogVLM2/datasets/V2/images/test"
    low_conf_save_dir = "outputs/low_confidence/YOLOv11n"
    output_csv_path = "/home/zy/1.Code/watermeter_YOLO-CogVLM2/outputs/V2/0602_YOLO+CogVLM2.csv"
    detailed_csv_path = "/home/zy/1.Code/watermeter_YOLO-CogVLM2/outputs/V2/0602_CogVLM2.csv"
    gt_csv_path = "/home/zy/1.Code/watermeter_YOLO-CogVLM2/outputs/V2/test.csv"  # 真实标签文件路径

    process_folder(folder_path, yolo_model, low_conf_save_dir, output_csv_path, detailed_csv_path, gt_csv_path)
