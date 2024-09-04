# yolov8_inference.py

from PIL import Image
import torch
import math
import os
import csv
import logging
import time
import numpy as np

import sys
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到sys.path
sys.path.append(project_root)

from utils.yolov8_utils import YOLOV8


logging.basicConfig(filename='test_yolov8_inference.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DigitalPredictor_Yolov8():
    def __init__(self, model=0):
        self.model = model
        self.yolo_predicter = YOLOV8(model_type='s')
    
    def call(self, img_path):
        logging.info(f"推理图片: {os.path.basename(img_path)}")
        try:
            # 读取图片
            start1_time = time.time()
            img = Image.open(img_path)
            # print(f"读取图片时间：{time.time() - start1_time}秒")
            logging.info(f"读取图片 {img_path} 时间 {time.time() - start1_time} 秒")
        except FileNotFoundError:
            # print("文件不存在，请检查文件路径是否正确")
            logging.error("文件不存在，请检查文件路径是否正确")
            return "file_not_found"
        except Exception as e:
            # print(f"读取图片时发生异常：{e}")
            logging.error(f"读取图片发生异常: {e}")
            return "read_image_error"
        
        try:
            start2_time = time.time()
            xyxyn,cls = self.yolo_predicter.predict(img)          
            logging.info(f"YOLOV8s检测时间：{time.time() - start2_time}秒")
        except Exception as e:
            logging.error(f"YOLOV8s检测数字框时发生异常: {e}")
            return "yolov8_detection_error"

        start3_time = time.time()
        #对返回来的进行判断，若出现下标是模糊的直接报错，先进行从左到右排序，
        # 根据第一个张量的第一列进行排序，并调整第二个张量的位置关系
        sorted_values, sorted_indices = torch.sort(xyxyn[:, 0])
        sorted_tensor2 = cls[sorted_indices]
        # print('sorted_tensor2',sorted_tensor2)
        # 检查元素是否存在
        element_to_find = 20
        contains_element = torch.any(sorted_tensor2 == element_to_find).item()

        if contains_element:
            #存在模糊字符，报错
            return 'error'
        else:
            #将下标替换成对应的真是数字
            
            # 定义变化规则
            conversion_rules = {
                0:0, 1:0.5, 2:1, 3:1.5, 4:2, 5:2.5,
                6:3, 7:3.5, 8:4, 9:4.5, 10:5,
                11:5.5, 12:6, 13:6.5, 14:7, 15:7.5,
                16:8, 17:8.5, 18:9, 19:9.5, 20:20
            }
            
         # 更新原始张量
        for i in range(len(sorted_tensor2)):
            original_value = sorted_tensor2[i].item()
            new_value = conversion_rules.get(original_value, original_value)
            sorted_tensor2[i] = new_value
        
        # 将不是最后一位数字的先进行向下取整再拼接
        concatenated_string = ''.join(str(math.floor(x.item())) for x in sorted_tensor2) # 沿着默认的维度0进行连接
        print(concatenated_string)
        if len(concatenated_string)<3:return 'error'
        logging.info(f"数据后处理时间{time.time() - start3_time}秒")
        logging.info(f"Result: {concatenated_string}")  
        
        return os.path.basename(img_path), concatenated_string

if __name__ == '__main__':
    digital_processor = DigitalPredictor_Yolov8()  
    
    images_folder = "/home/zy/1.Code/new_water_meter_recognition/test_images"
    total_time = 0
    total_images = 0
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]

    with open('test.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['image_name', 'results'])
        writer.writeheader()

        for image_file in os.listdir(images_folder):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                image_path = os.path.join(images_folder, image_file)
                
                start_time = time.time()
                image_name, result = digital_processor.call(image_path)
                end_time = time.time()
                total_time += end_time - start_time
                total_images += 1
                
                logging.info(f"推理图片所占时间 {os.path.basename(image_path)}: {end_time - start_time} seconds")
                
                writer.writerow({'image_name': image_name, 'results': result})

    if total_images > 0:
        average_time = total_time / total_images
        logging.info(f"总共推理图片数量: {total_images}")
        logging.info(f"完成所有图片推理总时间: {total_time} 秒")
        logging.info(f"平均推理一张图片所占时间: {average_time} 秒")
        print(f"总共推理图片数量: {total_images}")
        print(f"完成所有图片推理总时间: {total_time} 秒")
        print(f"平均推理一张图片所占时间: {average_time} 秒")
    else:
        print("文件夹中没有找到任何图片文件。")