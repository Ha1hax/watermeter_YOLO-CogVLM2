import pandas as pd

def calculate_optimized_mrrate(real_csv, pred_csv):
    # 读取真实和预测CSV，确保保留字符串格式
    real_df = pd.read_csv(real_csv, dtype=str)
    pred_df = pd.read_csv(pred_csv, dtype=str)

    # 过滤掉 '异常' 行
    pred_df = pred_df[~pred_df['results'].astype(str).str.startswith("异常")].copy()

    # 重命名列以对齐
    pred_df = pred_df.rename(columns={'results': 'Reading_pred'})
    # 注意：real_df列名取决于你的test CSV实际列名，调整为'YourColumnName'
    real_df = real_df.rename(columns={'Sequence Annotation': 'Reading_real'})

    # 合并，仅保留交集
    merged_df = pd.merge(real_df, pred_df, on="image_name", how="inner")

    total_meters = len(merged_df)
    match_count = 0

    for _, row in merged_df.iterrows():
        real_value_str = row['Reading_real']
        pred_value_str = row['Reading_pred']

        try:
            real_value = int(real_value_str)
            pred_value = int(pred_value_str)

            if abs(real_value - pred_value) <= 1:
                match_count += 1
        except ValueError:
            # 如果转换失败（非数字），跳过
            continue

    OMRrate = match_count / total_meters if total_meters > 0 else 0

    print(f"有效样本总数: {total_meters}")
    print(f"匹配样本数 (±1 容差): {match_count}")
    print(f"优化后的MRrate: {OMRrate:.2%}")

    return OMRrate

# 示例调用（请替换为你的真实路径）
real_csv = "outputs/V2/test.csv"  # 真实标签，仅测试集
pred_csv = "outputs/V2/0602_YOLOOnly.csv"  # YOLO预测结果
calculate_optimized_mrrate(real_csv, pred_csv)
