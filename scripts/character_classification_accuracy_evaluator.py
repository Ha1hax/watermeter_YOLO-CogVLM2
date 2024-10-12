import os
import csv
import logging
from PIL import Image
import sys

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到sys.path
sys.path.append(project_root)

from utils.yolo_utils import YOLOModel
from utils.cogvlm_inference import is_half_character

# 配置日志
log_file = "accuracy_classification.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),  # 日志写入文件
    logging.StreamHandler()  # 同时输出到控制台
])

def evaluate_accuracy(folder_path, yolo_model):
    """
    评估YOLOv11和CogVLM2在字符分类上的准确度。
    :param folder_path: 包含字符分类子文件夹的路径，每个子文件夹名称为真实标签。
    :param yolo_model: YOLOv11模型
    """
    yolo_correct = 0
    cogvlm_correct = 0
    total_images = 0

    # 打开 CSV 文件，保存每张图片的详细结果
    with open("accuracy_results.csv", mode='w', newline='') as csv_file:
        fieldnames = ['image_name', 'true_label', 'yolo_prediction', 'cogvlm_prediction', 'yolo_correct', 'cogvlm_correct']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # 遍历每个类别文件夹
        for true_label in os.listdir(folder_path):
            true_label_path = os.path.join(folder_path, true_label)
            if os.path.isdir(true_label_path):
                for image_name in os.listdir(true_label_path):
                    if image_name.endswith(('.jpg', '.png')):
                        image_path = os.path.join(true_label_path, image_name)
                        image = Image.open(image_path).convert('RGB')

                        # 使用YOLOv11进行推理
                        yolo_result = yolo_model.predict(image_path, conf_thres=0.3)

                        # 检查 YOLO 是否检测到任何对象
                        if hasattr(yolo_result[0], 'boxes') and yolo_result[0].boxes.cls.numel() > 0:
                            yolo_prediction = yolo_result[0].boxes.cls[0].item()
                        else:
                            yolo_prediction = "未知"
                            logging.info(f"图片 {image_name} 未检测到任何对象")

                        # 使用CogVLM2进行推理
                        cogvlm_result = is_half_character(image)
                        cogvlm_prediction = cogvlm_result["value"]

                        # 记录是否正确
                        yolo_is_correct = str(yolo_prediction) == true_label
                        cogvlm_is_correct = str(cogvlm_prediction) == true_label

                        # 更新统计
                        yolo_correct += int(yolo_is_correct)
                        cogvlm_correct += int(cogvlm_is_correct)
                        total_images += 1

                        # 写入CSV结果
                        writer.writerow({
                            'image_name': image_name,
                            'true_label': true_label,
                            'yolo_prediction': yolo_prediction,
                            'cogvlm_prediction': cogvlm_prediction,
                            'yolo_correct': yolo_is_correct,
                            'cogvlm_correct': cogvlm_is_correct
                        })

                        logging.info(f"处理图片 {image_name}，真实标签: {true_label}, YOLOv11预测: {yolo_prediction}, CogVLM2预测: {cogvlm_prediction}")

    # 计算准确度
    yolo_accuracy = yolo_correct / total_images if total_images > 0 else 0
    cogvlm_accuracy = cogvlm_correct / total_images if total_images > 0 else 0

    logging.info(f"YOLOv11 准确率: {yolo_accuracy * 100:.2f}%")
    logging.info(f"CogVLM2 准确率: {cogvlm_accuracy * 100:.2f}%")
    print(f"YOLOv11 准确率: {yolo_accuracy * 100:.2f}%")
    print(f"CogVLM2 准确率: {cogvlm_accuracy * 100:.2f}%")

    return yolo_accuracy, cogvlm_accuracy


if __name__ == "__main__":
    # 初始化 YOLOv8 模型
    yolo_model = YOLOModel(model_version='v11')  # 使用 YOLOv8 模型

    # 输入文件夹路径
    folder_path = "/home/zy/1.Code/NumsClassification/dataset/val"  # 替换为包含字符分类的图片文件夹路径

    # 评估准确度
    evaluate_accuracy(folder_path, yolo_model)
