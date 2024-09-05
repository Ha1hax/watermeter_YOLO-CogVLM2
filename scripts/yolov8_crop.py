import os
import cv2
import sys

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到sys.path
sys.path.append(project_root)

from utils.yolov8_utils import YOLOv8

def crop_and_save_objects(image_path, results, save_dir):
    """
    根据YOLOv8的预测结果裁剪每个检测到的对象并另存为新图片
    :param image_path: 原始图片路径
    :param results: YOLOv8推理结果
    :param save_dir: 保存裁剪结果的目录
    """
    # 加载原始图片
    img = cv2.imread(image_path)
    img_name = os.path.basename(image_path).split('.')[0]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历所有检测结果
    for i, result in enumerate(results):
        boxes = result.boxes.xyxy  # 获取边界框的坐标
        classes = result.boxes.cls  # 获取类别信息
        confidences = result.boxes.conf  # 获取置信度

        for j, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            # 裁剪对象
            x_min, y_min, x_max, y_max = map(int, box)
            cropped_img = img[y_min:y_max, x_min:x_max]

            # 保存裁剪的图片
            cropped_img_path = os.path.join(save_dir, f"{img_name}_object_{i}_{j}.jpg")
            cv2.imwrite(cropped_img_path, cropped_img)
            print(f"保存裁剪图片: {cropped_img_path}")


if __name__ == '__main__':
    # 图片和保存目录
    image_folder = '/home/zy/1.Code/new_water_meter_recognition/data/0-9_4-6'
    save_dir = '/home/zy/1.Code/new_water_meter_recognition/cropped_digits'
    
    # 初始化YOLOv8模型 (无需传递模型路径)
    yolo = YOLOv8()

    # 遍历图片并进行推理和裁剪
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)

        # 进行推理
        results = yolo.predict(img_path)

        # 保存裁剪结果
        crop_and_save_objects(img_path, results, save_dir)
