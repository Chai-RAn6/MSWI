import os
import random
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
########################################
# 1. 读取原始文件并构造多层邻接矩阵
########################################

def load_cerna_data(
        layer_file="pierreauger_layers.txt",
        edge_file="pierreauger_multiplex.edges",
        node_file="pierreauger_nodes.txt"
):
    """
    读取多层网络文件，返回：
      adjacency_list: [adj_matrix_of_layer1, ..., adj_matrix_of_layerN]
      n_layers: 层数
      n_nodes: 节点数
      node_id_map: 节点ID -> 名称 的映射
      layer_id_map: 层ID -> 名称 的映射
    """
    # 1) 读取 layer 信息
    layer_id_map = {}
    with open(layer_file, "r", encoding="utf-8") as f:
        header = next(f).strip().split()  # 跳过第一行: "layerID layerLabel"
        for line in f:
            line=line.strip()
            if not line:
                continue
            cols=line.split()
            lay_id = int(cols[0])
            layer_id_map[lay_id] = cols[1]  # 例如 1->"direct_interaction"
    n_layers = len(layer_id_map)

    # 2) 读取 node 信息
    node_id_map = {}
    with open(node_file, "r", encoding="utf-8") as f:
        header = next(f).strip().split()  # 读取表头 "nodeID nodeLabel"
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split()  # 使用split()自动处理空格
            node_id = int(cols[0])  # 解析节点ID
            node_label = cols[1]  # 解析节点标签
            node_id_map[node_id] = node_label
    n_nodes = max(node_id_map.keys())

    # 3) 初始化多层邻接矩阵(用0填充)
    adjacency_list = [np.zeros((n_nodes, n_nodes)) for _ in range(n_layers)]

    # 4) 读取 edges 并填充邻接矩阵
    #    假设格式: layerID nodeA nodeB weight
    #    若实际只有3列(无weight)，则需相应调整
    with open(edge_file, "r", encoding="utf-8") as f:
        next(f)  # 跳过表头
        lines = f.readlines()

    for line in tqdm(lines, desc="Reading edges"):
        line = line.strip()
        if not line:
            continue
        cols = line.split("\t")
        if len(cols) != 3:
            continue
        ndA, ndB, layID = map(int, cols)
        adjacency_list[layID][ndA, ndB] = 1
        adjacency_list[layID][ndB, ndA] = 1  # 无向图

    return adjacency_list, n_layers, n_nodes, node_id_map, layer_id_map, node_file

########################################
# 2. 计算 FWI 所需的核心函数
########################################

def calculate_degree_distance(k_i, k_j):
    if k_i==0 or k_j==0:
        return 0
    return max(k_i,k_j)/min(k_i,k_j)

def pseudo_inverse_laplacian(adj_matrix, eps=1e-9, rcond_val=1e-9):
    """
    对拉普拉斯矩阵的伪逆做一个数值稳定性处理：
    1) 在 Laplacian 矩阵上加一个微小扰动 eps * I
    2) 调用 np.linalg.pinv 时，设置 rcond=rcond_val
    """
    size = len(adj_matrix)
    degree_matrix = np.diag(adj_matrix.sum(axis=1))
    laplacian_matrix = degree_matrix - adj_matrix

    # 加微小扰动
    laplacian_matrix += np.eye(size) * eps

    # middle_term = Lp - (1/N)*ee^T
    ones = np.ones((size, 1))
    middle_term = laplacian_matrix - (1/size)*np.outer(ones, ones.T)
    # 用 pinv 计算伪逆
    middle_term_inv = np.linalg.pinv(middle_term, rcond=rcond_val)
    # 最终 pseudo-inverse
    pseudo_inv = middle_term_inv + (1/size)*np.outer(ones, ones.T)
    return pseudo_inv

def calculate_degree_distance(k_i, k_j):
    if k_i==0 or k_j==0:
        return 0
    return max(k_i,k_j)/min(k_i,k_j)

def calculate_pairwise_information(degree_distance, resistance, shortest_path):
    return (degree_distance**2)*resistance*shortest_path

def kl_divergence(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    return np.sum(X * np.log((X+1e-15)/(Y+1e-15)))

def jensen_shannon_divergence(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    M = 0.5*(X+Y)
    return 0.5*kl_divergence(X, M) + 0.5*kl_divergence(Y, M)

def compute_information_distribution(neighbor_nodes, degrees_dict):
    f = len(neighbor_nodes)
    V = np.zeros(f)
    total_deg = sum(degrees_dict[n] for n in neighbor_nodes)
    for idx, n in enumerate(neighbor_nodes):
        if total_deg>0:
            V[idx] = degrees_dict[n]/total_deg
    return V

def compute_layer_intralayer_info(adj_matrix, layer_idx=0):
    """
    对某一层（layer_idx可选仅做标识），计算每个节点 i 的层内信息:
      In_i^alpha = \sum_j ( (deg_dist^2)*resistance*shortestPath )
    并返回 (intralayer_info, degrees)
    """
    N = len(adj_matrix)
    degrees = adj_matrix.sum(axis=1)

    # -- 1. 计算度距离矩阵
    deg_dist_mat = np.zeros((N,N))
    print(f"  [Layer {layer_idx}] Building degree-distance matrix...")
    for i in tqdm(range(N), desc=f"    deg-dist layer{layer_idx}", leave=False):
        for j in range(N):
            if i != j:
                deg_dist_mat[i,j] = calculate_degree_distance(degrees[i], degrees[j])

    # -- 2. 计算最短路径
    print(f"  [Layer {layer_idx}] Computing shortest paths...")
    G = nx.from_numpy_array(adj_matrix)
    sp_mat = np.zeros((N,N))
    for i in tqdm(range(N), desc=f"    shortest-path layer{layer_idx}", leave=False):
        dist_dict = nx.single_source_shortest_path_length(G, i)
        for j,d in dist_dict.items():
            sp_mat[i,j] = d

    # -- 3. 拉普拉斯伪逆 + 有效电阻
    print(f"  [Layer {layer_idx}] Calculating Laplacian pseudo-inverse & resistance...")
    pseudoInv = pseudo_inverse_laplacian(adj_matrix)  # 正则化版本
    res_mat = np.zeros((N,N))
    for i in tqdm(range(N), desc=f"resistance layer{layer_idx}", leave=False):
        for j in range(N):
            if i != j:
                res_mat[i,j] = abs(pseudoInv[i,i] + pseudoInv[j,j] - 2*pseudoInv[i,j])

    # -- 4. 汇总每个节点 i 的 In_i^alpha
    print(f"  [Layer {layer_idx}] Summarizing Intralayer info...")
    intralayer_info = np.zeros(N)
    for i in tqdm(range(N), desc=f"    In-layer sum layer{layer_idx}", leave=False):
        val_i = 0
        for j in range(N):
            if j!=i:
                dd = deg_dist_mat[i,j]
                rr = res_mat[i,j]
                sp = sp_mat[i,j]
                val_i += calculate_pairwise_information(dd, rr, sp)
        intralayer_info[i] = val_i

    return intralayer_info, degrees

def compute_extra_layer_info_for_node(i, adjacency_list):
    """
    计算节点 i 在各层的 E_JS(i)^alpha
    返回: {alpha: E_JS_of_alpha}
    """
    n_layers = len(adjacency_list)
    N = len(adjacency_list[0])

    # 先把每层的邻居 + 度数都存起来
    layers_neighbors = {}
    layers_degs = {}
    for alpha in range(n_layers):
        mat = adjacency_list[alpha]
        neigh_list = np.where(mat[i]>0)[0].tolist()  # i 的邻居
        deg_dict = {}
        for nd in range(N):
            deg_dict[nd] = mat[nd].sum()
        layers_neighbors[alpha] = neigh_list
        layers_degs[alpha]      = deg_dict

    extra_dict = {}
    # 对每一层 alpha, 计算它与其余层的 JS 散度之和
    for alpha in range(n_layers):
        neighbors_alpha = set(layers_neighbors[alpha])
        degs_alpha      = layers_degs[alpha]

        # 构造 "邻居并集" 决定分布向量的长度
        # 也可以考虑一次性把 (alpha vs. 所有其他层) 的并集都合并，
        # 但文献公式(13) 是 pairwise 累加, 效果一样。
        # 这里做 alpha vs. beta
        sum_div_alpha = 0.0

        for beta in range(n_layers):
            if beta == alpha:
                continue
            neighbors_beta = set(layers_neighbors[beta])
            degs_beta      = layers_degs[beta]

            # 并集
            neighbors_union = neighbors_alpha.union(neighbors_beta)

            # 构造 V_alpha, V_beta (长度 = len(neighbors_union))
            # 按照文献(14): 如果节点 r 在该层邻居里，则 v = deg(r)/total_deg，否则 0
            # total_deg_alpha 是 alpha层的 sum_{r in alphaNeigh} deg(r)
            total_deg_alpha = sum(degs_alpha[r] for r in neighbors_alpha) or 1e-15
            total_deg_beta  = sum(degs_beta[r] for r in neighbors_beta)  or 1e-15

            V_alpha = []
            V_beta  = []

            for node_r in neighbors_union:
                # alpha 部分
                if node_r in neighbors_alpha:
                    V_alpha.append(degs_alpha[node_r]/total_deg_alpha)
                else:
                    V_alpha.append(0.0)

                # beta 部分
                if node_r in neighbors_beta:
                    V_beta.append(degs_beta[node_r]/total_deg_beta)
                else:
                    V_beta.append(0.0)

            # 现在 V_alpha, V_beta 形状相同
            js_val = jensen_shannon_divergence(V_alpha, V_beta)
            sum_div_alpha += js_val

        extra_dict[alpha] = sum_div_alpha

    return extra_dict

def compute_mld_weights_real(adjacency_list, max_s, t, include_self=False):
    """
    计算每个节点在每层的多局部维度 MLD，并归一化为 W_i^alpha。
    支持不同 π_i(t, s) 定义，支持是否排除自身节点。

    参数:
        adjacency_list: 多层网络的邻接矩阵列表
        max_s: 最大盒子大小（跳数）
        t: pi函数的类型（0, 1 或其他）
        include_self: 是否在 N_i(s) 中包含节点本身

    返回:
        all_weights[i] = {alpha: w}
    """

    n_layers = len(adjacency_list)
    n_nodes = len(adjacency_list[0])
    all_weights = []

    for i in tqdm(range(n_nodes), desc="Computing MLD"):
        mld_per_layer = []

        for alpha in range(n_layers):
            adj = adjacency_list[alpha]
            G = nx.from_numpy_array(adj)

            ln_s_vals = []
            ln_pi_vals = []

            for s in range(1, max_s + 1):
                sp_lengths = nx.single_source_shortest_path_length(G, i, cutoff=s)

                if include_self:
                    N_i_s = len(sp_lengths)
                else:
                    N_i_s = len(sp_lengths) - 1  # 不含自己

                mu_i_s = N_i_s / n_nodes

                # π_i(t,s) 计算
                if t == 0:
                    pi_i_s = 1.0 / (mu_i_s + 1e-15)
                elif t == 1:
                    pi_i_s = mu_i_s * np.log(mu_i_s + 1e-15)
                else:
                    pi_i_s = mu_i_s ** t

                ln_s_vals.append(np.log(s))
                ln_pi_vals.append(np.log(abs(pi_i_s) + 1e-15))

            slope, _ = np.polyfit(ln_s_vals, ln_pi_vals, deg=1)
            MLD_i_alpha = slope if t == 1 else slope / (t - 1)
            mld_per_layer.append(MLD_i_alpha)

        total = sum(mld_per_layer)
        w_dict = {alpha: mld_per_layer[alpha] / total if total > 0 else 0 for alpha in range(n_layers)}
        all_weights.append(w_dict)

    return all_weights

def compute_fwi_for_node(i, intralayer_info_list, extra_info_dict_list, weight_dict_i):
    """
    FWI_i = sum_alpha [ exp(- In_i^alpha * W_i^alpha) + exp(- E_JS(i)^alpha)*(1 - W_i^alpha ) ]
    """
    n_layers = len(intralayer_info_list)
    val=0
    for alpha in range(n_layers):
        In_i_alpha = intralayer_info_list[alpha][i]
        Ex_i_alpha = extra_info_dict_list[i][alpha]
        W_i_alpha  = weight_dict_i[alpha]
        term1 = np.exp( - In_i_alpha*W_i_alpha )
        term2 = np.exp( - Ex_i_alpha )*(1 - W_i_alpha)
        val  += term1 + term2
    return val

#输出为 CSV 文件
def save_fwi_to_csv(fwi_ranked, filename):
     df = pd.DataFrame(fwi_ranked, columns=["NodeID", "FWI", "NodeLabel"])
     df["Rank"] = range(1, len(df) + 1)  # 添加排名列
     df = df[["Rank", "NodeID", "NodeLabel", "FWI"]]  # 重排列顺序
     df.to_csv(filename, index=False, encoding="utf-8-sig")
     print(f"已将 FWI 排名结果保存为：{filename}")


def get_prefix_from_nodefile(node_file):
    basename = os.path.basename(node_file)              # e.g. "hcc_nodes.txt"
    prefix = basename.split("_")[0]                     # e.g. "hcc"
    return prefix


    # Step 1: 运行 main() 得到 FWI 值和节点排名
    main()

    # Step 2: 再次读取数据用于 Mf 批量测试
    adjacency_list, n_layers, n_nodes, node_map, layer_map, node_file = load_cerna_data()
    run_mf_batch_tests(adjacency_list, n_nodes, node_map, node_file)


def run_full_analysis(adjacency_list, n_nodes, node_map, node_file):
    from tqdm import tqdm
    from scipy.stats import rankdata
    import pandas as pd
    import numpy as np
    from collections import Counter
    import os


    def calculate_monotonicity(rankings):
        N = len(rankings)
        if N <= 1:
            return 0.0
        rank_counts = Counter(rankings)
        numerator = sum(nr * (nr - 1) for nr in rank_counts.values())
        denominator = N * (N - 1)
        Mf = (1 - numerator / denominator) ** 2
        return Mf

    def get_prefix_from_nodefile(node_file):
        basename = os.path.basename(node_file)
        prefix = basename.split("_")[0]
        return prefix

    configs = [{"t": 2, "exclude_self": False}]
    for idx, cfg in enumerate(configs):
        label = chr(65 + idx)
        print(f"\n组合 {label} ：t = {cfg['t']}, exclude_self = {cfg['exclude_self']}")

        # Step 1: 层内信息
        intralayer_info_list = []
        for alpha in range(len(adjacency_list)):
            info_arr, _ = compute_layer_intralayer_info(adjacency_list[alpha], layer_idx=alpha)
            intralayer_info_list.append(info_arr)

        # Step 2: 层间信息
        extra_info_list = []
        for i in tqdm(range(n_nodes), desc="Extra-layer"):
            e_dict = compute_extra_layer_info_for_node(i, adjacency_list)
            extra_info_list.append(e_dict)

        # Step 3: 权重（MLD）
        all_weights = compute_mld_weights_real(
            adjacency_list,
            max_s=5,
            t=cfg["t"],
            include_self=not cfg["exclude_self"]
        )

        # Step 4: FWI 值计算
        fwi_values = np.zeros(n_nodes)
        for i in tqdm(range(n_nodes), desc="FWI"):
            w_dict_i = all_weights[i]
            fwi_values[i] = (compute_fwi_for_node(i, intralayer_info_list, extra_info_list, w_dict_i)
                             + np.random.uniform(0, 1e-8))

        fwi_values = np.round(fwi_values, 10)
        rankings = rankdata(-fwi_values, method='min')
        Mf = calculate_monotonicity(rankings)
        print(f"✅ Mf = {Mf:.6f}")

        # Step 5: 输出前10与后10 FWI
        fwi_ranked = [(i + 1, fwi_values[i], node_map.get(i + 1, f"Node_{i+1}")) for i in range(n_nodes)]
        # Python list 排序（控制台）
        fwi_ranked.sort(key=lambda x: (-x[1], x[0]))  # 先按 FWI 降序，再按 NodeID 升序

        print("\n>>> 前 10 个 FWI 最高的节点:")
        for rank, (nodeID, fwi, nodeLabel) in enumerate(fwi_ranked[:10], start=1):
            print(f"Rank {rank}: Node {nodeID} ({nodeLabel}) - FWI = {fwi:.4f}")

        print("\n>>> 后 10 个 FWI 最低的节点:")
        for i, (nodeID, fwi, nodeLabel) in enumerate(fwi_ranked[-10:], start=n_nodes - 9):
            print(f"Rank {i}: Node {nodeID} ({nodeLabel}) - FWI = {fwi:.4f}")

        # Step 6: 保存结果
        df = pd.DataFrame({
            "NodeID": np.arange(1, n_nodes + 1),
            "FWI": fwi_values,
            "NodeLabel": [node_map.get(i + 1, f"Node_{i+1}") for i in range(n_nodes)],
            "Rank": rankings
        })
        prefix = get_prefix_from_nodefile(node_file)
        outname = f"fwi_{prefix}.csv"
        # Pandas 排序（保存 CSV）
        df = df.sort_values(by=["FWI", "NodeID"], ascending=[False, True]).reset_index(drop=True)

        df.to_csv(outname, index=False, encoding="utf-8-sig")
        print(f"✅ 已保存至：{outname}")
        print("分析完成 ✅")


if __name__ == "__main__":
    adjacency_list, n_layers, n_nodes, node_map, layer_map, node_file = load_cerna_data()

    run_full_analysis(adjacency_list, n_nodes, node_map, node_file)
