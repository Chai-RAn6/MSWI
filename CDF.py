import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression

def plot_real_cdf_with_fit(file_path, decimals=[6, 7, 8, 9], colors=["red", "green", "blue", "purple"]):
    """
    同时绘制多个保留小数位的CDF曲线，并对每条曲线线性拟合，输出斜率。
    """
    df = pd.read_csv(file_path)
    fwi_scores = df["FWI"].values
    total = len(fwi_scores)

    plt.figure(figsize=(8, 6))

    for i, d in enumerate(decimals):
        rounded_scores = np.round(fwi_scores, d)
        score_counter = Counter(rounded_scores)
        sorted_score_groups = sorted(score_counter.items(), key=lambda x: -x[0])

        x = []
        y = []
        cum = 0
        for j, (score, count) in enumerate(sorted_score_groups):
            cum += count
            x.append(j + 1)
            y.append(1 - cum / total)

        # 拟合斜率
        # x_array = np.array(x).reshape(-1, 1)
        # y_array = np.array(y)
        # model = LinearRegression().fit(x_array, y_array)
        # slope = model.coef_[0]

        # 画 CDF 曲线
        plt.step(x, y, where="post", label=f"FWI (round={d})\n", color=colors[i], linewidth=2)

        # 拟合线（用虚线画出）
        # y_fit = model.predict(x_array)
        # plt.plot(x, y_fit, linestyle="--", color=colors[i], alpha=0.6)

    # 图形设置
    plt.xlabel("Rank Group (by rounded FWI)", fontsize=14)
    plt.ylabel("CDF", fontsize=14)
    plt.title("CDF Curve with Linear Fit", fontsize=16)
    plt.grid(True)
    plt.xlim(left=0)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 调用函数
plot_real_cdf_with_fit("fwi_pierreauger.csv")
