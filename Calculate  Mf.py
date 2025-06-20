import pandas as pd
from collections import Counter
from scipy.stats import rankdata

def calculate_monotonicity(rankings):
    """
    根据排名列表计算 Mf 值（单调性指标）

    参数:
        rankings (list): 节点排名结果，例如 [1, 2, 2, 3, 4]
    返回:
        float: Mf 值，范围在 [0, 1]，越接近1表示区分度越高
    """
    N = len(rankings)
    if N <= 1:
        return 0.0  # 无法区分

    # 统计每个排名的出现次数
    rank_counts = Counter(rankings)
    numerator = sum(nr * (nr - 1) for nr in rank_counts.values())
    denominator = N * (N - 1)

    Mf = (1 - numerator / denominator) ** 2
    return Mf

def compute_mf_from_csv(csv_path):
    """
    从CSV文件中读取 FWI 排名并计算 Mf 指标
    """
    df = pd.read_csv(csv_path)

    # 获取 FWI 列，并生成从高到低的排名（高分排前）
    fwi_scores = df["FWI"].values
    rankings = rankdata(-fwi_scores, method='min')  # min: 相同值分配相同最小名次

    # 计算 Mf 值
    Mf_value = calculate_monotonicity(rankings)

    # 输出
    print(f"数据集的 Mf 值为: {Mf_value:.6f}")
    return Mf_value

# 🚀 执行计算
compute_mf_from_csv("fwi_pierreauger.csv")
