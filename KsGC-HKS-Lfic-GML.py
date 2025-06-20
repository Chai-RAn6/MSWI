import pandas as pd
import networkx as nx
import math
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor

# ========== 参数 ==========
EDGE_FILE = "pierreauger_overlap_multiplex.edges"
OUTPUT_FILE = "PMC_node_scores.csv"

# ========== 图构建 ==========
def load_graph(edge_file):
    # 尝试读取数据
    try:
        df = pd.read_csv(edge_file, sep=None, engine='python', header=None)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None

    # 自动判断是否包含 header
    if df.shape[1] < 2:
        raise ValueError("❌ 边文件列数不足，至少需要 source 和 target 两列。")

    # 如果有 2 列，假设为 source 和 target
    if df.shape[1] == 2:
        df.columns = ["source", "target"]
    # 如果有 3 列或更多，默认前两列为 source 和 target
    else:
        df.columns = ["source", "target"] + [f"col{i}" for i in range(2, df.shape[1])]

    # 创建图
    G = nx.Graph()
    for _, row in df.iterrows():
        try:
            G.add_edge(int(row["source"]), int(row["target"]))
        except ValueError as ve:
            print(f"⚠️ 跳过无效行: {row}，原因: {ve}")
            continue

    # 移除自环
    G.remove_edges_from(nx.selfloop_edges(G))

    print(f"✅ 图加载完成，共有 {G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边。")
    return G


# ========== KsGC ==========
def ks_gravity_centrality(G):
    k_shells = nx.core_number(G)
    return {
        node: k_shells[node] * (G.degree(node) ** 0.5)
        for node in G.nodes()
    }

# ========== HKS ==========
def hierarchical_k_shell(G):
    k_shells = nx.core_number(G)
    hks = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        neighbor_shells = [k_shells[n] for n in neighbors] if neighbors else [0]
        hks[node] = k_shells[node] + sum(neighbor_shells) / len(neighbor_shells)
    return hks

# ========== LFIC ==========
def information_entropy(G, node):
    neighbors = list(G.neighbors(node))
    degree = len(neighbors)
    entropy = 0
    for neighbor in neighbors:
        if degree == 0: continue
        p = G.degree(neighbor) / degree
        entropy -= p * math.log2(p) if p > 0 else 0
    return entropy

def lfic_centrality(G):
    centrality = {}
    for node in G.nodes():
        entropy = information_entropy(G, node)
        neighbor_entropy = sum(information_entropy(G, n) for n in G.neighbors(node)) if G.degree(node) > 0 else 0
        centrality[node] = entropy + neighbor_entropy
    return centrality

# ========== GML ==========
def extract_node_features(G):
    features = []
    node_list = []
    for node in G.nodes():
        degree = G.degree(node)
        clustering = nx.clustering(G, node)
        neighbors = list(G.neighbors(node))
        avg_neighbor_degree = np.mean([G.degree(n) for n in neighbors]) if neighbors else 0
        features.append([degree, clustering, avg_neighbor_degree])
        node_list.append(node)
    return np.array(features), node_list

def gml_node_scoring(G):
    X, node_list = extract_node_features(G)
    y = np.sum(X, axis=1)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    scores = model.predict(X)
    return dict(zip(node_list, scores))

# ========== Mf 计算 ==========
def calculate_monotonicity(rankings):
    N = len(rankings)
    if N <= 1:
        return 0.0
    from collections import Counter
    rank_counts = Counter(rankings)
    numerator = sum(nr * (nr - 1) for nr in rank_counts.values())
    denominator = N * (N - 1)
    Mf = (1 - numerator / denominator) ** 2
    return Mf

# ========== 主流程 ==========
if __name__ == "__main__":
    print("读取网络...")
    G = load_graph(EDGE_FILE)
    df_results = pd.DataFrame()
    df_results["NodeID"] = list(G.nodes())

    print("计算 KsGC...")
    ksgc = ks_gravity_centrality(G)
    df_results["KsGC"] = df_results["NodeID"].map(ksgc)

    print("计算 HKS...")
    hks = hierarchical_k_shell(G)
    df_results["HKS"] = df_results["NodeID"].map(hks)

    print("计算 LFIC...")
    lfic = lfic_centrality(G)
    df_results["LFIC"] = df_results["NodeID"].map(lfic)

    print("计算 GML...")
    gml = gml_node_scoring(G)
    df_results["GML"] = df_results["NodeID"].map(gml)

    print("保存节点得分至文件...")
        # 保存节点得分至文件（保留 10 位小数）
    df_results = df_results.round(6)
    df_results.to_csv(OUTPUT_FILE, index=False)

    # 添加扰动 + 计算 Mf
    epsilon = 1e-16
    print("\n各方法的 Mf 值（保留 6 位 + 加扰动）:")
    for method in ["KsGC", "HKS", "LFIC", "GML"]:
        # 添加扰动打破并列
        scores = df_results[method].round(6) + epsilon * df_results["NodeID"]
        rankings = rankdata(-scores, method="min")  # 越大越靠前
        mf = calculate_monotonicity(rankings)
        print(f"{method}: Mf = {mf:.6f}")

