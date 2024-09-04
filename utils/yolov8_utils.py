# from ultralytics import YOLO, RTDETR
# import torch

# class YOLOV8:
#     def __init__(self, model_type='s'):
#         if model_type == 'n':
#             self.model = YOLO('/home/zy/1.Code/watermeter/models/yolov8/weights_yolov8_n/best.pt')
#         elif model_type == 's':
#             self.model = YOLO('/home/zy/1.Code/new_water_meter_recognition/models/0827_yolov8s/best.pt')
#         elif model_type == 'm':
#             self.model = YOLO('/home/zy/1.Code/watermeter/models/yolov8/weights_yolov8_m/best.pt')
#         elif model_type == 'rt_detr':
#             self.model = RTDETR('/home/zy/1.Code/watermeter/models/yolov8/weights_yolov8_s/best.pt')

#     def predict(self, img):
#         results = self.model(img)
#         xyxyn, cls = [], []

#         for result in results:
#             boxes = result.boxes  # 边界框输出的 Boxes 对象
#             xyxyn = boxes.xyxyn  # 提取归一化的xyxy坐标
#             cls = boxes.cls  # 提取类别标签

#         return xyxyn, cls


# if __name__ == "__main__":
#     yolo = YOLOV8(model_type='s')
#     xyxy_results = yolo.predict('/home/zy/0.Data/watermeter/test_100/911004002018_20240222.jpg')
#     print('output', xyxy_results)
from ultralytics import YOLO, RTDETR
import torch

class YOLOV8:
    def __init__(self, model_type='s'):
        if model_type == 'n':
            self.model = YOLO('/home/zy/1.Code/watermeter/models/yolov8/weights_yolov8_n/best.pt')
        elif model_type == 's':
            self.model = YOLO('/home/zy/1.Code/new_water_meter_recognition/models/0829_yolov8s/best.pt')
            # self.model = YOLO('/home/zy/1.Code/new_water_meter_recognition/models/yolov8/weights_yolov8_s/best.pt')
        elif model_type == 'm':
            self.model = YOLO('/home/zy/1.Code/watermeter/models/yolov8/weights_yolov8_m/best.pt')
        elif model_type == 'rt_detr':
            self.model = RTDETR('/home/zy/1.Code/watermeter/models/yolov8/weights_yolov8_s/best.pt')

    def predict(self, img):
        results = self.model(img)
        xyxyn, cls, confs = [], [], []

        for result in results:
            boxes = result.boxes  # 边界框输出的 Boxes 对象
            xyxyn = boxes.xyxyn  # 提取归一化的xyxy坐标
            cls = boxes.cls  # 提取类别标签
            confs = boxes.conf  # 提取置信度分数

        return xyxyn, cls, confs  # 返回置信度分数


if __name__ == "__main__":
    yolo = YOLOV8(model_type='s')
    xyxy_results = yolo.predict('/home/zy/0.Data/watermeter/test_100/911004002018_20240222.jpg')
    print('output', xyxy_results)
