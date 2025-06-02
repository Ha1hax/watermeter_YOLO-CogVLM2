# utils/detection_model_utils.py

"""
detection_model_utils.py
========================
目标检测模型加载与推理工具类。
支持 YOLOv8/v11/v12 (基于ultralytics YOLO) 和 RT-DETR。
"""
import os
import logging
from ultralytics import YOLO
import torch

class Detector:
    def __init__(self, model_version='yolov11', model_path=None, device='cuda'):
        """
        初始化目标检测模型。
        :param model_version: 模型版本标识 ('yolov8' / 'yolov11' / 'yolov12' / 'rtdetr')
        :param model_path: 自定义模型权重路径（可选）
        :param device: 推理设备，默认为 'cuda'
        """
        self.model_version = model_version.lower()
        self.device = device

        # 默认模型路径字典（按需修改）
        default_paths = {
            'yolov8': './models/yolov8n.pt',
            'yolov11': '/home/zy/1.Code/watermeter_YOLO-CogVLM2/models/V2/dx/yolo11n.pt',
            'yolov12': './models/yolov12n.pt',
            'rtdetr': './models/rtdetr.pth'
        }

        # 确定模型路径
        if model_path:
            self.model_path = model_path
        else:
            if self.model_version not in default_paths:
                raise ValueError(f"不支持的模型版本: {self.model_version}")
            self.model_path = default_paths[self.model_version]

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件未找到: {self.model_path}")

        logging.info(f"加载模型 [{self.model_version}]，路径: {self.model_path}，设备: {self.device}")

        # 加载模型
        if self.model_version.startswith('yolo'):
            self.model = YOLO(self.model_path)
        elif self.model_version == 'rtdetr':
            self.model = torch.load(self.model_path, map_location=device)
            self.model.eval()
        else:
            raise ValueError(f"不支持的模型版本: {self.model_version}")

    def predict(self, image_path, conf_thres=0.5):
        """
        进行目标检测预测。
        :param image_path: 图片路径
        :param conf_thres: 置信度阈值
        :return: (boxes, scores, classes) -> 检测框, 置信度, 类别
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件未找到: {image_path}")

        logging.info(f"使用 [{self.model_version}] 推理图片: {image_path}，置信度阈值: {conf_thres}")

        if self.model_version.startswith('yolo'):
            # 使用 YOLO 模型推理
            results = self.model.predict(image_path, conf=conf_thres)
            r = results[0]  # 只处理一张图片

            # 从结果中提取检测框、置信度、类别
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else []
            scores = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else []
            classes = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else []

            return boxes, scores, classes

        elif self.model_version == 'rtdetr':
            from torchvision import transforms
            from PIL import Image

            img = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((640, 640)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)

            pred_logits = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            pred_boxes = outputs['pred_boxes'][0]

            threshold = conf_thres
            keep = pred_logits.max(-1).values > threshold
            boxes = pred_boxes[keep].cpu().numpy()
            classes = pred_logits[keep].argmax(-1).cpu().numpy()
            scores = pred_logits[keep].max(-1).values.cpu().numpy()

            return boxes, scores, classes

        else:
            raise ValueError(f"不支持的模型版本: {self.model_version}")
