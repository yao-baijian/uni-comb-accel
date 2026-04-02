import networkx as nx
import metis
import os

def load_gset_to_networkx(filename):
    """从 Gset 文件加载图到 NetworkX，并处理权重。"""
    G = nx.Graph()
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) < 1:
        return G
    
    n_vertices, n_edges = map(int, lines[0].split())
    
    # 添加顶点
    G.add_nodes_from(range(n_vertices))
    
    # 添加边和权重
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            u = int(parts[0]) - 1  # 转换为0-based索引
            v = int(parts[1]) - 1
            # Gset格式：第三列是权重（整数或浮点数）
            weight = int(float(parts[2])) if len(parts) >= 3 else 1
            G.add_edge(u, v, weight=weight)
    
    # 关键：设置图属性，告诉 METIS 使用 'weight' 作为边权重属性
    G.graph['edge_weight_attr'] = 'weight'
    
    return G

def partition_gset_with_metis_balanced(filename, nparts=2, imbalance_tolerance=0.04):
    """
    使用 METIS 对 Gset 图进行分区，控制不平衡度
    """
    G = load_gset_to_networkx(filename)
    n_vertices = G.number_of_nodes()
    
    # 计算允许的最大分区大小差
    max_imbalance_nodes = int(n_vertices * imbalance_tolerance)
    
    # 设置 ubvec 参数：负载不平衡容忍度
    ubvec_value = 1.0 + imbalance_tolerance
    
    # 调用 metis.part_graph 进行分区
    (edgecuts, parts) = metis.part_graph(
        G, 
        nparts, 
        contig=True,
        ubvec=[ubvec_value]  # 不平衡容忍度向量
    )
    
    return edgecuts, parts, G, max_imbalance_nodes

def calculate_actual_cut_weight(G, parts):
    """计算实际的割边权重和"""
    actual_cut_weight = 0
    for u, v, data in G.edges(data=True):
        if parts[u] != parts[v]:
            actual_cut_weight += data.get('weight', 1)
    return actual_cut_weight

# 主程序 - 测试不同不平衡容忍度
graph_files = [f'G{i}' for i in range(1, 51)]
data_dir = './data/Gset/'

# 测试不同的不平衡容忍度
imbalance_levels = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]

for imbalance_tol in imbalance_levels:
    print(f"\n{'='*100}")
    print(f"不平衡容忍度: {imbalance_tol*100:.1f}%")
    print(f"{'='*100}")
    print(f"{'Graph':<6} | {'METIS Cut':<10} | {'Actual Cut':<12} | {'Size0':<8} | {'Size1':<8} | "
          f"{'|Diff|':<8} | {'Max Allowed':<12} | {'Status':<10}")
    print("-" * 100)
    
    for i in range(1, 50):  # 先测试前10个图
        graph_file = f'G{i}'
        file_path = os.path.join(data_dir, graph_file)
        
        if not os.path.exists(file_path):
            continue
        
        try:
            # 分区（带不平衡约束）
            edgecut, parts, G, max_allowed = partition_gset_with_metis_balanced(
                file_path, 
                nparts=2, 
                imbalance_tolerance=imbalance_tol
            )
            
            # 计算分区大小
            size0 = parts.count(0)
            size1 = parts.count(1)
            size_diff = abs(size1 - size0)
            
            # 检查约束
            meets_constraint = size_diff <= max_allowed
            status = "OK" if meets_constraint else "VIOLATED"
            
            # 计算实际割边权重和
            actual_cut = calculate_actual_cut_weight(G, parts)
            
            # 表格化输出
            print(f"{graph_file:<6} | {edgecut:<10} | {actual_cut:<12} | {size0:<8} | {size1:<8} | "
                  f"{size_diff:<8} | {max_allowed:<12} | {status:<10}")
            
        except Exception as e:
            print(f"{graph_file:<6} | ERROR: {str(e)[:20]}")
