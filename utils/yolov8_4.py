
from ultralytics import YOLO, RTDETR
from PIL import Image
import os
import torch
import math


class YOLOV8():
    def __init__(self, model_type) -> None:
        if model_type == 'n':
            self.model = YOLO('/home/zy/1.Code/watermeter/models/yolov8/weights_yolov8_n/best.pt')
        elif model_type == 's':
            self.model = YOLO('/home/zy/1.Code/watermeter/models/yolov8/weights_yolov8_s/best.pt')
        elif model_type == 'm':
            self.model = YOLO('/home/zy/1.Code/watermeter/models/yolov8/weights_yolov8_m/best.pt')
        elif model_type == 'rt_detr':
            self.model = RTDETR('/home/zy/1.Code/watermeter/models/yolov8/weights_yolov8_s/best.pt')

    # 以后可以批量实现读数
    def predict(self, img) :
        # 在图片列表上运行批量推理
        # 返回 Results 对象列表, verbose 关闭输出详情
        #黑白对调
        # imgpath=str(img)
        # with Image.open(img) as img:

        # img_path=img.filename
        # img = img.convert("L")
        
        # inverted_img = Image.eval(img, lambda x: 255 - x)
        # img.paste(inverted_img, (0, 0))
        # inverted_img.save(img_path)
        results = self.model(img)
        # print(results)
        # 处理结果列表
        sorted_tensor2=[]

        for result in results:
            boxes = result.boxes  # 边界框输出的 Boxes 对象
          
            # print('boxes------------:',boxes)
            
            # print('xyxyn----',boxes.xyxyn)
            xyxyn=boxes.xyxyn
            # print('cls---------',boxes.cls)
            cls=boxes.cls
            # print('len(xyxyn)',len(xyxyn))
            # # 根据第一个张量的第一列进行排序，并调整第二个张量的位置关系
            # sorted_values, sorted_indices = torch.sort(xyxyn[:, 0])
            # print(len(sorted_indices))
            # # sorted_tensor1 = xyxyn[sorted_indices]
            # sorted_tensor2 = cls[sorted_indices]
            # print('sorted_tensor2',sorted_tensor2)
            # # 将张量中的元素连接起来
            # concatenated_string = ''.join(str(math.floor(x.item())) for x in sorted_tensor2) # 沿着默认的维度0进行连接
            # print("concatenated_string:")
            # print(concatenated_string)



        # debug 用
        # print('YOLOV8  debug')
        # im_array = results[0].plot(labels=False, conf=True)  # 绘制包含预测结果的BGR numpy数组
        # im_array = results[0].plot( conf=True)  # 绘制包含预测结果的BGR numpy数组
        # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
        # im.save('results.jpg')  # 保存图像
        # outpath='/home/chuanzhi/zy/water-meter-detection/outputs/1128/w_b'
        # image_name=os.path.join(outpath,os.path.basename(str(img_path)))
        # print(image_name)
        # im.save(image_name)  # 保存图像
        return xyxyn,cls
        '''
        关于 YOLO V8 返回值的理解 https://docs.ultralytics.com/zh/modes/predict/#boxes
        '''


if __name__ == "__main__":
    # yolo = YOLOV8(model_type='rt_detr')
    yolo = YOLOV8(model_type='s')
    xyxy_results = yolo.predict(
        '/home/zy/1.Code/new_water_meter_recognition/test_images/911004003261_20240110.jpg')
    print('output',xyxy_results)
    
\

    # folder_path = "/home/chuanzhi/zy/test_data/1124/pic"
    # files = [os.path.abspath(os.path.join(folder_path, f)) for f in os.listdir(folder_path)]
    # i =0
    # count =0
    # for file in files:
    #     xyxy_results = yolo.predict(file)
    #     i +=1
    #     if len(xyxy_results)>0:
    #         print(xyxy_results)
    #         count +=1
    #         print('xyxy_reslut',len(xyxy_results))
    #     print('图片能识别到框的数量count',count)
        
    


