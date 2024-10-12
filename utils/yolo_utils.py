# from ultralytics import YOLO
# import os
# import logging

# class YOLOv8:
#     def __init__(self, model_path=None, device='cuda'):
#         """
#         初始化YOLOv8模型，模型路径在类内定义
#         :param model_path: 训练好的模型路径，如果不传则使用默认路径
#         :param device: 运行设备 (默认cuda, 如果没有GPU则使用cpu)
#         """
#         # 定义模型路径，如果没有传入则使用默认路径
#         self.model_path = model_path or '/home/zy/1.Code/new_water_meter_recognition/models/0829_yolov8s/best.pt'
#         self.device = device
        
#         # 检查模型文件是否存在
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(f"模型文件未找到: {self.model_path}")
        
#         logging.info(f"加载模型: {self.model_path}，设备: {self.device}")
        
#         # 加载本地训练的YOLOv8模型权重
#         self.model = YOLO(self.model_path)  # 直接加载模型文件

#     def predict(self, image_path, conf_thres=0.5):
#         """
#         使用YOLOv8进行推理预测
#         :param image_path: 待推理的图片路径
#         :param conf_thres: 置信度阈值 (默认0.5)
#         :return: 推理结果
#         """
#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"图片文件未找到: {image_path}")
        
#         logging.info(f"正在推理图片: {image_path}，置信度阈值: {conf_thres}")
        
#         # 使用 YOLOv8 进行推理
#         results = self.model.predict(image_path, conf=conf_thres)
        
#         # logging.info(f"推理完成，检测到 {len(results)} 个对象")
        
#         return results

#     def save_results(self, results, save_dir):
#         """
#         保存推理结果
#         :param results: YOLOv8推理结果
#         :param save_dir: 保存结果的目录
#         """
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
        
#         # logging.info(f"保存推理结果到: {save_dir}")
#         results.save(save_dir)
from ultralytics import YOLO
import os
import logging

class YOLOModel:
    def __init__(self, model_version='v8', model_path=None, device='cuda'):
        """
        初始化YOLO模型，支持多版本模型（如YOLOv8，YOLOv11）。
        :param model_version: 模型版本 (默认 'v8')，可以选择 'v8' 或 'v11'
        :param model_path: 训练好的模型路径，如果不传则使用默认路径
        :param device: 运行设备 (默认 'cuda', 如果没有GPU则使用 'cpu')
        """
        self.model_version = model_version
        # 根据模型版本选择默认路径
        if model_path is None:
            if self.model_version == 'v8':
                self.model_path = '/home/zy/1.Code/new_water_meter_recognition/models/yolov8n/best.pt'
            elif self.model_version == 'v11':
                self.model_path = '/home/zy/1.Code/watermeter_YOLO-CogVLM2/models/weights/best.pt'
            else:
                raise ValueError(f"不支持的模型版本: {self.model_version}")
        else:
            self.model_path = model_path
        
        self.device = device
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件未找到: {self.model_path}")
        
        logging.info(f"加载 {self.model_version} 模型: {self.model_path}，设备: {self.device}")
        
        # 加载指定版本的YOLO模型
        self.model = YOLO(self.model_path)  # 直接加载模型文件

    def predict(self, image_path, conf_thres=0.5):
        """
        使用指定版本的YOLO模型进行推理预测
        :param image_path: 待推理的图片路径
        :param conf_thres: 置信度阈值 (默认 0.5)
        :return: 推理结果
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件未找到: {image_path}")
        
        logging.info(f"正在使用 {self.model_version} 推理图片: {image_path}，置信度阈值: {conf_thres}")
        
        # 使用 YOLO 进行推理
        results = self.model.predict(image_path, conf=conf_thres)
        
        # logging.info(f"推理完成，检测到 {len(results)} 个对象")
        
        return results

    def save_results(self, results, save_dir):
        """
        保存推理结果
        :param results: YOLO推理结果
        :param save_dir: 保存结果的目录
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # logging.info(f"保存推理结果到: {save_dir}")
        results.save(save_dir)
