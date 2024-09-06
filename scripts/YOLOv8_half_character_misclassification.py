import os
import cv2
import logging
import sys
from PIL import Image

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到sys.path
sys.path.append(project_root)

from utils.yolov8_utils import YOLOv8
from utils.cogvlm_inference import is_half_character

# 配置日志
log_file = "yolov8_half_character_misclass_4-6_09.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),  # 日志写入文件
    logging.StreamHandler()  # 同时输出到控制台
])

# 定义YOLOv8类别到数值的映射
yolo_class_map = {
    0: 0, 1: 0.5, 2: 1, 3: 1.5, 4: 2, 5: 2.5,
    6: 3, 7: 3.5, 8: 4, 9: 4.5, 10: 5,
    11: 5.5, 12: 6, 13: 6.5, 14: 7, 15: 7.5,
    16: 8, 17: 8.5, 18: 9, 19: 9.5
}

def is_misclassification(yolo_cls, cogvlm_value):
    """
    判断是否为YOLOv8将半字符错检成整数类别的情况。
    :param yolo_cls: YOLOv8检测出的类别
    :param cogvlm_value: CogVLM2检测出的值
    :return: True 表示YOLOv8将半字符错检为整数，False 否则
    """
    yolo_value = yolo_class_map[int(yolo_cls)]
    if isinstance(cogvlm_value, str) and '.' in cogvlm_value:
        cogvlm_value = float(cogvlm_value)

    # 检查是否差值为0.5，并且YOLOv8的值为整数
    if isinstance(cogvlm_value, float) and abs(yolo_value - cogvlm_value) == 0.5:
        return True
    return False

def process_image(image_path, yolo_model, save_dir, low_conf_threshold=0.9):
    """
    处理单张图片并判断YOLOv8是否将半字符误检为整数类别
    :param image_path: 图片路径
    :param yolo_model: YOLOv8模型
    :param save_dir: 错误检测图片保存的目录
    :param low_conf_threshold: YOLOv8置信度阈值
    :return: 是否有错检情况，返回True表示错检，False表示没有错检
    """
    logging.info(f"开始处理图片: {image_path}")

    # 1. 使用 YOLOv8 进行推理
    results = yolo_model.predict(image_path)

    if hasattr(results[0], 'boxes'):
        boxes = results[0].boxes.xyxy  # 获取检测框坐标
        confidences = results[0].boxes.conf  # 获取置信度
        classes = results[0].boxes.cls  # 获取类别

        logging.info(f"检测到的框: {boxes}")
        logging.info(f"置信度: {confidences}")
        logging.info(f"类别: {classes}")
    else:
        logging.error(f"YOLOv8 推理结果不包含 'boxes' 属性，请检查推理输出结构。")
        return False

    # 检查检测框数量
    if len(boxes) < 4:
        logging.error(f"图片 {image_path} 中的检测框数量少于 4，返回异常值。")
        return False

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
        return False

    misclassifications = 0
    image_misclassified = False

    # 3. 针对每个框使用 CogVLM2 进行推理
    for i, box in enumerate(boxes):
        conf = confidences[i]  # YOLOv8的置信度
        cls = int(classes[i])  # YOLOv8 类别

        if 0.6 <= conf < low_conf_threshold:
            # 裁剪出数字框
            x_min, y_min, x_max, y_max = map(int, box[:4])
            digit_image = cv2.imread(image_path)[y_min:y_max, x_min:x_max]

            # 将 `numpy.ndarray` 转换为 `PIL.Image`
            digit_image_pil = Image.fromarray(cv2.cvtColor(digit_image, cv2.COLOR_BGR2RGB))

            # 使用 CogVLM2 进行推理
            digit_result = is_half_character(digit_image_pil)

            logging.info(f"YOLOv8 类别: {cls}, 置信度: {conf}, CogVLM2 结果: {digit_result}")

            # 判断是否为误检
            if is_misclassification(cls, digit_result["value"]):
                misclassifications += 1
                image_misclassified = True
                logging.warning(f"图片 {image_path} 中，YOLOv8 将半字符错检为整数类别。")

    # 如果存在误检，将图片保存到指定目录
    if image_misclassified:
        save_misclassified_image(image_path, save_dir)

    return misclassifications

def save_misclassified_image(image_path, save_dir):
    """
    将含有YOLOv8错检为整数的半字符的图片另存到指定文件夹
    :param image_path: 原始图片路径
    :param save_dir: 错误检测图片保存的目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, image_name)
    cv2.imwrite(save_path, cv2.imread(image_path))
    logging.info(f"保存错检图片到 {save_path}")

def process_folder(folder_path, yolo_model, save_dir, low_conf_threshold=0.9):
    """
    处理文件夹中的所有图片，并统计YOLOv8将半字符错检成整数类别的数量
    :param folder_path: 图片文件夹路径
    :param yolo_model: YOLOv8模型
    :param save_dir: 错误检测图片保存的目录
    :param low_conf_threshold: YOLOv8置信度阈值
    """
    total_misclassifications = 0
    total_images = 0

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        if img_path.endswith(('.jpg', '.png')):
            total_images += 1
            misclassifications = process_image(img_path, yolo_model, save_dir, low_conf_threshold)
            total_misclassifications += misclassifications

    logging.info(f"文件夹 {folder_path} 中，总共处理了 {total_images} 张图片，半字符错检为整数类别的数量为 {total_misclassifications}。")

if __name__ == "__main__":
    # 初始化 YOLOv8 模型
    yolo_model = YOLOv8()  # 使用 YOLOv8 模型

    # 输入图片文件夹路径
    folder_path = "/home/zy/1.Code/new_water_meter_recognition/data/0-9_4-6"  # 替换为你的图片文件夹路径
    save_dir = "/home/zy/1.Code/new_water_meter_recognition/data/misclassified_images>0.6"  # 替换为保存错检图片的路径

    # 处理文件夹
    process_folder(folder_path, yolo_model, save_dir)
