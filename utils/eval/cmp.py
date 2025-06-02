import pandas as pd

def compare_csv_files(real_csv_path, inferred_csv_path, output_csv_path):
    # 读取CSV文件，确保字符串格式
    real_data = pd.read_csv(real_csv_path, dtype=str)
    inferred_data = pd.read_csv(inferred_csv_path, dtype=str)

    # 过滤异常行
    inferred_data = inferred_data[~inferred_data['results'].astype(str).str.startswith("异常")].copy()

    # 重命名列
    inferred_data = inferred_data.rename(columns={'results': 'Reading_pred'})
    real_data = real_data.rename(columns={'Sequence Annotation': 'Reading_real'})

    # 合并
    merged_data = real_data.merge(inferred_data, on='image_name', how='inner')

    # 差值计算（以整数差为准，保留前导零不影响）
    def compute_difference(row):
        try:
            real_val = int(row['Reading_real'])
            pred_val = int(row['Reading_pred'])
            return abs(real_val - pred_val)
        except:
            return None

    merged_data['difference'] = merged_data.apply(compute_difference, axis=1)

    # 统计
    exact_matches = merged_data['difference'].eq(0).sum()
    diff_one_count = merged_data['difference'].eq(1).sum()

    # 差值大于1的记录
    large_diff_data = merged_data[merged_data['difference'] > 1]

    # 保存
    large_diff_data.to_csv(output_csv_path, index=False)

    print(f"有效样本总数: {len(merged_data)}")
    print(f"完全匹配的记录数量: {exact_matches}")
    print(f"差值为1的记录数量: {diff_one_count}")
    print(f"差值大于1的记录已保存到: {output_csv_path}")

# 示例用法
real_csv_path = "outputs/V2/test.csv"  # 替换为真实读数的CSV文件路径
inferred_csv_path = "outputs/V2/0602_YOLOOnly.csv"  # 替换为推理结果的CSV文件路径
large_diff_output_csv_path = "outputs/V2/large_diff.csv"  # 替换为差值大于1的记录保存路径

compare_csv_files(real_csv_path, inferred_csv_path, large_diff_output_csv_path)
