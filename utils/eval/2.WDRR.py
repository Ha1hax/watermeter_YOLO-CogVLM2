import pandas as pd
from math import exp

def calculate_optimized_wdrr(real_csv, pred_csv):
    """
    计算优化后的权重感知容差字符级别识别率 (WDRR)
    """
    # 读取真实读数和预测读数，保留字符串格式
    real_df = pd.read_csv(real_csv, dtype=str)
    pred_df = pd.read_csv(pred_csv, dtype=str)

    # 过滤掉 '异常' 行
    pred_df = pred_df[~pred_df['results'].astype(str).str.startswith("异常")].copy()

    # 重命名列以对齐
    pred_df = pred_df.rename(columns={'results': 'Reading_pred'})
    real_df = real_df.rename(columns={'Sequence Annotation': 'Reading_real'})

    # 合并数据集
    merged_df = pd.merge(real_df, pred_df, on="image_name", how="inner")

    total_meters = len(merged_df)
    total_score = 0

    for _, row in merged_df.iterrows():
        real_reading = str(row['Reading_real'])
        pred_reading = str(row['Reading_pred'])

        # 长度不一致惩罚
        length_penalty = abs(len(real_reading) - len(pred_reading)) / max(len(real_reading), len(pred_reading))

        # 动态权重分配
        max_len = max(len(real_reading), len(pred_reading))
        weight_sum = sum(exp(-(max_len - i - 1)) for i in range(max_len))
        weights = [exp(-(max_len - i - 1)) / weight_sum for i in range(max_len)]

        # 字符级别比较
        char_score = 0
        for i in range(max_len):
            real_char = real_reading[i] if i < len(real_reading) else None
            pred_char = pred_reading[i] if i < len(pred_reading) else None

            if real_char is None:  # 漏检
                char_score -= weights[i] * 2.0
            elif pred_char is None:  # 多检
                char_score -= weights[i] * 1.5
            elif real_char != pred_char:  # 错检
                char_score -= weights[i] * 1.0

        total_score += max(0, 1 - length_penalty + char_score)  # 总得分不能低于0

    WDRR = total_score / total_meters if total_meters > 0 else 0

    print(f"有效样本总数: {total_meters}")
    print(f"优化后的WDRR: {WDRR:.2%}")

    return WDRR

# 示例调用
real_csv = "outputs/V2/test.csv"  # 真实标签
pred_csv = "outputs/V2/0602_YOLOOnly.csv"  # YOLO预测结果
calculate_optimized_wdrr(real_csv, pred_csv)
