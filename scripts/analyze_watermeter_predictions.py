import pandas as pd
import re
from collections import defaultdict

# 用户配置路径
log_file_path = "/home/zy/1.Code/watermeter_YOLO-CogVLM2/V2-0602.log"
true_csv_path = "/home/zy/1.Code/watermeter_YOLO-CogVLM2/outputs/V2/test.csv"
output_csv_path = "./analysis_result.csv"
output_excel_path = "./analysis_result.xlsx"

# 读取真实读数
true_df = pd.read_csv(true_csv_path)
true_dict = dict(zip(true_df['image_name'], true_df['Sequence Annotation'].astype(str)))

# 解析日志文件
with open(log_file_path, 'r') as f:
    log_lines = f.readlines()

results = []
current_image = None
current_prediction = []
yolo_data = []

for line in log_lines:
    line = line.strip()
    if "开始处理图片:" in line:
        if current_image:
            results.append({
                "image_name": current_image,
                "yolo_cogvlm": yolo_data,
                "final_prediction": current_prediction
            })
            yolo_data = []
            current_prediction = []
        current_image = line.split(":")[-1].split("/")[-1].replace(".jpg", "")
    elif "YOLO 类别" in line and "CogVLM2 结果" in line:
        yolo_cls = re.search(r'YOLO 类别: (\d+)', line).group(1)
        conf = float(re.search(r'置信度: ([0-9.]+)', line).group(1))
        cog_value_match = re.search(r"'value': (.+?)\}", line)
        cog_half_match = re.search(r"'is_half_character': (.+?),", line)
        cog_value = cog_value_match.group(1).replace("'", "").replace("\"", "").strip() if cog_value_match else ""
        cog_half = cog_half_match.group(1) if cog_half_match else "False"
        yolo_data.append({
            "YOLO_cls": yolo_cls,
            "YOLO_conf": conf,
            "CogVLM2_value": cog_value,
            "CogVLM2_is_half": cog_half
        })
    elif "最终水表读数结果:" in line:
        pred = line.split(":")[-1].strip()
        current_prediction = pred

# 加入最后一张图片
if current_image:
    results.append({
        "image_name": current_image,
        "yolo_cogvlm": yolo_data,
        "final_prediction": current_prediction
    })

# 分析逐字符结果
analysis_data = []
for item in results:
    img_name = item["image_name"]
    true_seq = true_dict.get(img_name, None)
    pred_seq = item["final_prediction"]
    yolo_cogvlm = item["yolo_cogvlm"]
    true_seq_chars = list(true_seq) if true_seq else []
    pred_seq_chars = list(pred_seq) if isinstance(pred_seq, str) else []

    for i, det in enumerate(yolo_cogvlm):
        true_char = true_seq_chars[i] if i < len(true_seq_chars) else None
        pred_char = det["CogVLM2_value"]
        is_half = det["CogVLM2_is_half"]
        conf = det["YOLO_conf"]
        error_type = []

        # 判断错误类型
        if true_char != pred_char:
            error_type.append("字符错误")
        if conf < 0.5:
            error_type.append("低置信度")
        if is_half == "True":
            error_type.append("半字符")
        if "(" in pred_char or "," in pred_char:
            error_type.append("异常值")
        if true_char is None:
            error_type.append("长度超出")

        error_tag = ";".join(error_type) if error_type else "正确"

        analysis_data.append({
            "图片名": img_name,
            "字符位置": i + 1,
            "真实字符": true_char,
            "预测字符": pred_char,
            "是否半字符": is_half,
            "YOLO置信度": conf,
            "错误类型": error_tag
        })

# 保存分析结果
df_analysis = pd.DataFrame(analysis_data)
df_analysis.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
df_analysis.to_excel(output_excel_path, index=False)

print(f"分析完成！结果已保存为 {output_csv_path} 和 {output_excel_path}")
