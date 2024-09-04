import os
import sys
import time
import logging
import numpy as np
from PIL import Image

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到sys.path
sys.path.append(project_root)

from utils.yolov8_utils import YOLOV8
from utils.cogvlm_inference import is_half_character

# 日志配置
logging.basicConfig(level=logging.INFO)

def preprocess_boxes(xyxy, cls, img_size):
    # 将YOLO预测的边框格式进行转换为[顶部, 左侧, 底部, 右侧]
    boxes = xyxy[:, [1, 0, 3, 2]].cpu().numpy()  # 将tensor从GPU移到CPU，然后转为numpy数组
    
    # 按照x_min (左侧)进行排序，确保顺序从左到右
    sorted_indices = np.argsort(boxes[:, 1])
    boxes = boxes[sorted_indices]
    cls = cls[sorted_indices]  # 对 cls 也进行相同的排序

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

    return boxes, cls, areas, avg_area


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

def process_result(cls, cogvlm2_result):
    # 定义汉字数字与阿拉伯数字的映射
    chinese_to_digit = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, 
        "六": 6, "七": 7, "八": 8, "九": 9, "零": 0,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, 
        "6": 6, "7": 7, "8": 8, "9": 9, "0": 0
    }

    # 检查CogVLM2的输出是否包含多个数字特征
    digit_count = 1  # 默认认为是一个数字特征
    extracted_digits = []
    if '两个' in cogvlm2_result or '2' in cogvlm2_result or '二' in cogvlm2_result:
        digit_count = 2  # 识别到两个数字特征
        # 提取CogVLM2推理出的两个数字特征
        for char in cogvlm2_result:
            if char in chinese_to_digit:
                extracted_digits.append(chinese_to_digit[char])
                
    # 将YOLOv8的类别映射到整数和半字符
    yolo_class_map = {
        0: 0, 1: 0.5, 2: 1, 3: 1.5, 4: 2, 5: 2.5,
        6: 3, 7: 3.5, 8: 4, 9: 4.5, 10: 5,
        11: 5.5, 12: 6, 13: 6.5, 14: 7, 15: 7.5,
        16: 8, 17: 8.5, 18: 9, 19: 9.5
    }
    
    yolo_predicted_value = yolo_class_map[int(cls)]

    # 决策逻辑
    if isinstance(yolo_predicted_value, int) or (isinstance(yolo_predicted_value, float) and yolo_predicted_value.is_integer()):
        # 如果CogVLM2识别到一个数字特征且YOLOv8检测为整数
        if digit_count == 1:
            return False, yolo_predicted_value
    if digit_count == 2 and not isinstance(yolo_predicted_value, int):
        # 如果CogVLM2识别到两个数字特征且YOLOv8检测为半字符
        return True, yolo_predicted_value
    elif digit_count == 2 and isinstance(yolo_predicted_value, int):
        # YOLOv8检测为整数但CogVLM2识别到两个数字特征，认为YOLOv8漏检了半字符
        # 根据CogVLM2的推理结果组合半字符值
        if len(extracted_digits) == 2:
            half_character_value = (extracted_digits[0] + extracted_digits[1]) / 2
            return True, half_character_value
        else:
            logging.warning(f"CogVLM2推理出两个数字特征，但未能提取出具体数字，类别：{cls}，结果：{cogvlm2_result}")
            return True, yolo_predicted_value  # 暂时返回半字符处理
    elif digit_count == 1 and not isinstance(yolo_predicted_value, int):
        # YOLOv8检测为半字符但CogVLM2只识别到一个数字特征
        logging.warning(f"YOLOv8 检测为半字符类别，但 CogVLM2 只识别到一个数字特征，类别：{cls}，结果：{cogvlm2_result}")
        return False, yolo_predicted_value  # 返回整数处理，实际情况很少见
    else:
        logging.error(f"无法匹配的决策逻辑，类别：{cls}，结果：{cogvlm2_result}")
        return False, None  # 无法处理的情况，返回 None


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

    # try:
    #     # 预处理检测到的框
    #     boxes, areas, avg_area = preprocess_boxes(xyxy, img_size)
    # except ValueError as e:
    #     logging.error(f"预处理时发生错误: {str(e)}")
    #     return str(e)
# 调整 recognize_water_meter 函数以反映上述更改
    try:
        # 预处理检测到的框
        boxes, cls, areas, avg_area = preprocess_boxes(xyxy, cls, img_size)
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

    final_reading = ""
    for i, cropped_img_path in enumerate(cropped_img_paths):
        logging.info(f"处理第 {i+1} 个裁剪图像，YOLOv8预测类别：{cls[i]}")
        
        try:
            with Image.open(cropped_img_path) as cropped_img:
                # 确保图像是RGB格式
                cropped_img = cropped_img.convert('RGB')
                cogvlm2_result = is_half_character(cropped_img)
            logging.info(f"CogVLM2推理结果：{cogvlm2_result}")
        except Exception as e:
            logging.error(f"CogVLM2推理时发生异常: {e}")
            return "cogvlm2_inference_error"

        is_half, predicted_value = process_result(cls[i], cogvlm2_result)
        
        if is_half:
            final_reading += f"{int(predicted_value)}.5"
        else:
            final_reading += f"{int(predicted_value)}"

    logging.info(f"最终的水表读数: {final_reading}")
    return final_reading

if __name__ == "__main__":
    img_path = "/home/zy/1.Code/new_water_meter_recognition/test_images/911004002069_20240601.jpg"
    recognize_water_meter(img_path)