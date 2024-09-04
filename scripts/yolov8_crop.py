import sys
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到sys.path
sys.path.append(project_root)

from utils.yolov8_utils import YOLOV8
import time
import logging
import numpy as np
from PIL import Image


def preprocess_boxes(xyxy, img_size):
    # 将YOLO预测的边框格式进行转换为[顶部, 左侧, 底部, 右侧]
    boxes = xyxy[:, [1, 0, 3, 2]].cpu().numpy()  # 先将tensor从GPU移到CPU，然后转为numpy数组
    
    # 按照x_min (左侧)进行排序，确保顺序从左到右
    boxes = boxes[np.argsort(boxes[:, 1])]

    # 计算每个检测框的水平间距、长度和面积
    gaps = [boxes[i + 1][1] - boxes[i][3] for i in range(len(boxes) - 1)]
    longs = [boxes[i][3] - boxes[i][1] for i in range(len(boxes))]
    areas = [(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in range(len(boxes))]
    
    # 计算除第一个检测框外的平均面积
    try:
        avg_area = sum(areas[1:]) / len(areas[1:])
    except ZeroDivisionError:
        raise ValueError("未检测到数字框")
    
    min_long = min(longs) if longs else 0
    
    # 检查间距是否大于最小长度
    for gap in gaps:
        if gap > min_long:
            raise ValueError("检测框之间的gap大于最小长度，可能不是水表上的字符框")

    return boxes, areas, avg_area


def crop_digits(image_path, boxes, areas, avg_area, img_size, save_dir='cropped_digits'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img = Image.open(image_path)
    cropped_images = []
    cropped_img_paths = []

    for i, box in enumerate(boxes):
        # 如果第一个检测框的面积小于平均面积的80%，则跳过
        if i == 0 and areas[0] < avg_area * 0.8:
            continue
        top, left, bottom, right = box
        top = max(0, np.floor(top * img_size[1]).astype('int32'))
        left = max(0, np.floor(left * img_size[0]).astype('int32'))
        bottom = min(img.size[1], np.floor(bottom * img_size[1]).astype('int32'))
        right = min(img.size[0], np.floor(right * img_size[0]).astype('int32'))

        # 裁剪并保存图像
        crop_image = img.crop([left, top, right, bottom])
        cropped_images.append(crop_image)

        cropped_img_path = os.path.join(save_dir, f'cropped_{i}.png')
        crop_image.save(cropped_img_path)
        cropped_img_paths.append(cropped_img_path)

        logging.info(f"保存裁剪图像: {cropped_img_path}, 尺寸: {crop_image.size}")

    return cropped_images, cropped_img_paths

def recognize_water_meter(img_path):
    logging.info(f"推理图片: {os.path.basename(img_path)}")
    try:
        # 读取图片
        start1_time = time.time()
        img = Image.open(img_path)
        img_size = img.size
        logging.info(f"读取图片 {os.path.basename(img_path)} 时间：{time.time()- start1_time}秒")
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
        xyxy, cls = yolo.predict(img)
        logging.info(f"YOLOv8检测时间：{time.time() - start2_time}秒")
    except Exception as e:
        logging.error(f"YOLOv8检测数字框时发生异常: {e}")
        return "yolov8_detection_error"

    try:
        # 预处理检测到的框
        boxes, areas, avg_area = preprocess_boxes(xyxy, img_size)
    except ValueError as e:
        logging.error(f"预处理时发生错误: {str(e)}")
        return str(e)

    try:
        # 裁剪数字框
        cropped_images, cropped_img_paths = crop_digits(img_path, boxes, areas, avg_area, img_size)
        logging.info(f"裁剪后的图像数量：{len(cropped_images)}")
    except Exception as e:
        logging.error(f"裁剪数字框时发生异常: {e}")
        return "crop_error"

    # 进一步处理裁剪后的图像
    # 这里可以调用 CogVLM2 模型进行进一步判断
    
    return cropped_images

if __name__ == "__main__":
    img_path = "/home/zy/1.Code/new_water_meter_recognition/test_images/911004002069_20240601.jpg"
    recognize_water_meter(img_path)
