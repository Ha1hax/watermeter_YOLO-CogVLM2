import pandas as pd
from Levenshtein import distance as levenshtein_distance

def calculate_metrics(real_csv, pred_csv):
    # 读取真实读数和预测读数
    real_df = pd.read_csv(real_csv)
    pred_df = pd.read_csv(pred_csv)

    # 合并数据集
    merged_df = pd.merge(real_df, pred_df, on="image_name", how="inner")

    # 初始化指标
    total_meters = len(merged_df)
    match_count = 0
    levenshtein_errors = []
    absolute_errors = []

    for _, row in merged_df.iterrows():
        # 转换为字符串
        real_reading = str(row['Reading_real'])
        pred_reading = str(row['Reading_pred'])

        # 仪表识别率 (MRrate)
        if real_reading == pred_reading:
            match_count += 1

        # 刻度识别率 (DRrate)
        lev_error = levenshtein_distance(real_reading, pred_reading)
        max_len = max(len(real_reading), len(pred_reading))
        levenshtein_errors.append(1 - (lev_error / max_len))

        # 平均绝对误差 (MAerror)
        try:
            real_value = int(real_reading)
            pred_value = int(pred_reading)
            absolute_errors.append(abs(real_value - pred_value))
        except ValueError:
            # 跳过非整数的情况
            absolute_errors.append(float('inf'))

    # 计算最终指标
    MRrate = match_count / total_meters
    DRrate = sum(levenshtein_errors) / total_meters
    MAerror = sum(absolute_errors) / total_meters

    return MRrate, DRrate, MAerror

# 使用示例
real_csv = 'data/GT.csv'  # 替换为真实文件路径
pred_csv = 'data/yolov3n.csv'  # 替换为预测文件路径
MRrate, DRrate, MAerror = calculate_metrics(real_csv, pred_csv)

print(f"仪表识别率 (MRrate): {MRrate:.2%}")
print(f"刻度识别率 (DRrate): {DRrate:.2%}")
print(f"平均绝对误差 (MAerror): {MAerror:.2f}")
