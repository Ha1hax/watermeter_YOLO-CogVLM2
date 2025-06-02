# import sys
# import os

# # 获取当前脚本的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 获取项目根目录
# project_root = os.path.dirname(current_dir)

# # 将项目根目录添加到sys.path
# sys.path.append(project_root)

# import os
# import torch
# from PIL import Image
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # 模型和设备配置
# MODEL_PATH = "/home/zy/1.Code/new_water_meter_recognition/models/cogvlm2-llama3-chinese-chat-19B-int4"
# TORCH_TYPE = torch.bfloat16
# device = 'cuda:0'

# # 加载标记器和模型
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=TORCH_TYPE, trust_remote_code=True, device_map=device).eval()

# def recur_move_to(item, tgt, criterion_func):
#     if criterion_func(item):
#         device_copy = item.to(tgt)
#         return device_copy
#     elif isinstance(item, list):
#         return [recur_move_to(v, tgt, criterion_func) for v in item]
#     elif isinstance(item, tuple):
#         return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
#     elif isinstance(item, dict):
#         return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
#     else:
#         return item

# def collate_fn(features, tokenizer) -> dict:
#     images = [feature.pop('images', None) for feature in features if 'images' in feature]
#     tokenizer.pad_token = tokenizer.eos_token
#     max_length = max(len(feature['input_ids']) for feature in features)

#     def pad_to_max_length(feature, max_length):
#         padding_length = max_length - len(feature['input_ids'])
#         feature['input_ids'] = torch.cat([feature['input_ids'], torch.full((padding_length,), tokenizer.pad_token_id)])
#         feature['token_type_ids'] = torch.cat([feature['token_type_ids'], torch.zeros(padding_length, dtype=torch.long)])
#         feature['attention_mask'] = torch.cat([feature['attention_mask'], torch.zeros(padding_length, dtype=torch.long)])
#         if feature['labels'] is not None:
#             feature['labels'] = torch.cat([feature['labels'], torch.full((padding_length,), tokenizer.pad_token_id)])
#         else:
#             feature['labels'] = torch.full((max_length,), tokenizer.pad_token_id)
#         return feature

#     features = [pad_to_max_length(feature, max_length) for feature in features]
#     batch = {
#         key: torch.stack([feature[key] for feature in features])
#         for key in features[0].keys()
#     }

#     if images:
#         batch['images'] = images

#     return batch
# # '''
# # 用于测试的函数，判断图像中是否有半字符。接受一个图像路径作为输入，返回一个字符串，表示是否有半字符。
# # '''
# # def is_half_character(image_path):
# #     query = '这张图片上有多少个数字的特征？'
# #     image = Image.open(image_path).convert('RGB')
# #     input_sample = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image], template_version='chat')
    
# #     input_batch = collate_fn([input_sample], tokenizer)
# #     input_batch = recur_move_to(input_batch, device, lambda x: isinstance(x, torch.Tensor))
# #     input_batch = recur_move_to(input_batch, torch.bfloat16, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))

# #     with torch.no_grad():
# #         outputs = model.generate(**input_batch, max_new_tokens=2048, pad_token_id=128002, top_k=1)
# #         outputs = outputs[:, input_batch['input_ids'].shape[1]:]
# #         result = tokenizer.batch_decode(outputs)[0].strip()

# #     return result

# '''
# 正式函数的接收一个PIL.Image对象作为输入，返回一个字符串，表示是否有半字符。
# '''
# def is_half_character(image):
#     # query = '这张图片上有多少个数字的特征？'
#     query = '这张图片上是数字几？'
#     input_sample = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image], template_version='chat')
    
#     input_batch = collate_fn([input_sample], tokenizer)
#     input_batch = recur_move_to(input_batch, device, lambda x: isinstance(x, torch.Tensor))
#     input_batch = recur_move_to(input_batch, torch.bfloat16, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))

#     with torch.no_grad():
#         outputs = model.generate(**input_batch, max_new_tokens=2048, pad_token_id=128002, top_k=1)
#         outputs = outputs[:, input_batch['input_ids'].shape[1]:]
#         result = tokenizer.batch_decode(outputs)[0].strip()

#     return result


# if __name__ == "__main__":
#     # 测试代码，直接运行脚本时执行
#     test_image_path = '/home/zy/1.Code/new_water_meter_recognition/cropped_digits/cropped_4.png'
    
#     # 将图像路径转换为 PIL.Image 对象
#     image = Image.open(test_image_path).convert('RGB')
    
#     # 调用 is_half_character 函数，传入 PIL.Image 对象
#     result = is_half_character(image)
#     print(f'图像 {test_image_path} 的半字符判断结果: {result}')

import sys
import os
import re
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到sys.path
sys.path.append(project_root)

# 模型和设备配置
MODEL_PATH = "/home/zy/.cache/modelscope/hub/ZhipuAI/cogvlm2-llama3-chinese-chat-19B-int4"
TORCH_TYPE = torch.bfloat16
device = 'cuda:0'

# 加载标记器和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=TORCH_TYPE, trust_remote_code=True, device_map=device).eval()

def recur_move_to(item, tgt, criterion_func):
    if criterion_func(item):
        device_copy = item.to(tgt)
        return device_copy
    elif isinstance(item, list):
        return [recur_move_to(v, tgt, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item

def collate_fn(features, tokenizer) -> dict:
    images = [feature.pop('images', None) for feature in features if 'images' in feature]
    tokenizer.pad_token = tokenizer.eos_token
    max_length = max(len(feature['input_ids']) for feature in features)

    def pad_to_max_length(feature, max_length):
        padding_length = max_length - len(feature['input_ids'])
        feature['input_ids'] = torch.cat([feature['input_ids'], torch.full((padding_length,), tokenizer.pad_token_id)])
        feature['token_type_ids'] = torch.cat([feature['token_type_ids'], torch.zeros(padding_length, dtype=torch.long)])
        feature['attention_mask'] = torch.cat([feature['attention_mask'], torch.zeros(padding_length, dtype=torch.long)])
        if feature['labels'] is not None:
            feature['labels'] = torch.cat([feature['labels'], torch.full((padding_length,), tokenizer.pad_token_id)])
        else:
            feature['labels'] = torch.full((max_length,), tokenizer.pad_token_id)
        return feature

    features = [pad_to_max_length(feature, max_length) for feature in features]
    batch = {
        key: torch.stack([feature[key] for feature in features])
        for key in features[0].keys()
    }

    if images:
        batch['images'] = images

    return batch

def clean_cogvlm_output(result):
    """
    清理 CogVLM2 返回的描述，移除无关的部分，保留有意义的数值部分。
    """
    # 去除 "<|end_of_text|>" 等无关部分，并删除多余的非数字字符
    result = result.replace("<|end_of_text|>", "").strip()
    
    # 去除所有非数字和 "和" 字符
    result = re.sub(r'[^\d和]', '', result)  # 仅保留数字和 "和"
    
    # print(f"清理后的结果: {result}")  # 打印清理后的结果，确保清理正确
    return result
def postprocess_cogvlm_result(result):
    """
    处理 CogVLM2 返回的描述，提取数值并判断是否为半字符。
    返回两个信息：
    1. 是否为过渡状态（半字符）。
    2. 返回的数值。
    """
    # 清理返回结果
    result = clean_cogvlm_output(result)
    
    # 查找所有数字
    match = re.findall(r'\d+', result)
    # print(f"提取到的数字: {match}")  # 打印提取到的数字，确保数字提取正常
    
    if not match:
        # print(f"未能提取出数值: {result}")
        return {"is_half_character": False, "value": "无法识别"}
    
    # 如果找到的数字是两位数，拆分成两个单独的数字
    if len(match) == 1 and len(match[0]) == 2:
        first_digit = int(match[0][0])  # 拆分第一个数字
        second_digit = int(match[0][1])  # 拆分第二个数字
        # print(f"first_digit: {first_digit}, second_digit: {second_digit}")  # 打印调试信息
        
        # 判断是否为特殊过渡态：如果是 "90"，表示 9.5
        if first_digit == 9 and second_digit == 0:
            return {"is_half_character": True, "value": "9.5"}
        
        # 判断是否为连续的过渡态
        if second_digit == first_digit + 1:
            return {"is_half_character": True, "value": f"{first_digit}.5"}
        else:
            # 返回不是过渡态，提供两个数字供后续 YOLOv8 推理判断
            return {"is_half_character": False, "value": (first_digit, second_digit)}

    # 如果找到两个数字，判断它们是否为过渡态
    elif len(match) == 2:
        first_digit = int(match[0])
        second_digit = int(match[1])
        # print(f"first_digit: {first_digit}, second_digit: {second_digit}")  # 打印调试信息
        
        # 特殊情况：如果第一个数字是 9，第二个是 0，表示 9.5
        if first_digit == 9 and second_digit == 0:
            return {"is_half_character": True, "value": "9.5"}
        
        # 判断是否为连续的过渡态（如 "7 和 8" 或 "78"）
        if second_digit == first_digit + 1:
            return {"is_half_character": True, "value": f"{first_digit}.5"}
        
        # 如果不是过渡态，返回两个数值供 YOLOv8 使用
        return {"is_half_character": False, "value": (first_digit, second_digit)}

    # 处理单个数字的情况
    elif len(match) == 1:
        return {"is_half_character": False, "value": match[0]}
    
    else:
        # print(f"无法解析的输出: {result}")
        return {"is_half_character": False, "value": "无法识别"}


def is_half_character(image):
    """用于推理是否是半字符的函数"""
    query = '这张图片上是数字几？'
    input_sample = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image], template_version='chat')
    
    input_batch = collate_fn([input_sample], tokenizer)
    input_batch = recur_move_to(input_batch, device, lambda x: isinstance(x, torch.Tensor))
    input_batch = recur_move_to(input_batch, torch.bfloat16, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))

    with torch.no_grad():
        outputs = model.generate(**input_batch, max_new_tokens=2048, pad_token_id=128002, top_k=1)
        outputs = outputs[:, input_batch['input_ids'].shape[1]:]
        result = tokenizer.batch_decode(outputs)[0].strip()

        # print(f"CogVLM2 返回的原始结果: {result}")

    # 调用 postprocess_cogvlm_result 函数处理返回的结果
    processed_result = postprocess_cogvlm_result(result)

    return processed_result


if __name__ == "__main__":
    # 测试代码，直接运行脚本时执行
    test_image_path = '/home/zy/1.Code/watermeter_YOLO-CogVLM2/clsdata/condition1/test/0.5/911004002311_20240506_202_10.jpg'
    
    # 将图像路径转换为 PIL.Image 对象
    image = Image.open(test_image_path).convert('RGB')
    
    # 调用 is_half_character 函数，传入 PIL.Image 对象
    result = is_half_character(image)
    print(f'图像 {test_image_path} 的半字符判断结果: {result}')
