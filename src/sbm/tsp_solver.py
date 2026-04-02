import numpy as np
import math
from typing import Tuple, List

from src.sbm.problem_to_ising import tsp_to_hamiltonian

_current_distances = None

def set_current_distance_matrix(distances: np.ndarray):
    global _current_distances
    _current_distances = distances

def get_current_distance_matrix() -> np.ndarray:
    return _current_distances

def extract_tsp_solution(spin_config: np.ndarray, N: int) -> List[int]:
    spin_matrix = spin_config.reshape(N, N)
    path = [-1] * N

    for j in range(N):
        for i in range(N):
            if spin_matrix[i, j] > 0:  # spin = +1
                # if path[j] != -1:
                    # print(f"Warning: position {j}  overlapped!")
                path[j] = i
    
    # 检查是否有缺失的位置
    # if -1 in path:
    #     print(f"Warning: path incomplete: {path}")
    
    return path

def calculate_path_distance(path: List[int], distances: np.ndarray) -> float:
    total = 0.0
    n = len(path)
    
    for i in range(n):
        total += distances[path[i], path[(i + 1) % n]]
    
    return total

def is_valid_tsp_solution(spin_config: np.ndarray, N: int) -> Tuple[bool, int, int]:
    spin_matrix = spin_config.reshape(N, N)
    
    row_sums = np.sum(spin_matrix > 0, axis=1)
    row_valid = np.all(row_sums == 1)
    
    col_sums = np.sum(spin_matrix > 0, axis=0)
    col_valid = np.all(col_sums == 1)
    
    overlap_count = np.sum(row_sums > 1)
    missing_count = np.sum(col_sums == 0)
    
    return (row_valid and col_valid), overlap_count, missing_count

def read_tsplib_data(filename: str) -> Tuple[int, np.ndarray, str]:
    coordinates = []
    dimension = 0
    name = ""
    reading_coords = False
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            if line.startswith("NAME"):
                name = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            elif line.startswith("EOF"):
                break
            elif reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    # 忽略第一列（节点编号），读取坐标
                    x, y = float(parts[1]), float(parts[2])
                    coordinates.append([x, y])
    
    return dimension, np.array(coordinates), name

def bsb_tsp(J: np.ndarray, num_iters: int = 10000, dt: float = 0.1, 
                     constraint_strength: float = 5.0) -> Tuple[List[float], List[float], np.ndarray]:
    N = J.shape[0]
    x_comp = np.random.randn(N) * 0.01
    y_comp = np.random.randn(N) * 0.01
    xi = 0.5 / math.sqrt(N)
    
    energies = []
    alpha = np.linspace(0, 1, num_iters)
    
    for i in range(num_iters):
        y_comp += ((-1 + alpha[i]) * x_comp + xi * (J @ x_comp)) * dt
        x_comp += y_comp * dt
        y_comp[np.abs(x_comp) > 1] = 0.
        x_comp = np.clip(x_comp, -1, 1)
        # 应用TSP约束投影
        # if i % 10 == 0:  # 每10次迭代应用一次约束
        #     x_comp = project_tsp_constraints(x_comp, city_count, constraint_strength)
        
        sol = np.sign(x_comp)
        energy = -0.5 * sol.T @ J @ sol
        
        energies.append(energy)
    
    return energies, np.sign(x_comp)

def project_tsp_constraints(x_comp: np.ndarray, city_count: int, strength: float) -> np.ndarray:
    N = city_count
    x_new = x_comp.copy().reshape(N, N)
    
    for j in range(N):
        col = x_new[:, j]
        softmax = np.exp(strength * col)
        softmax = softmax / np.sum(softmax)
        x_new[:, j] = 2 * softmax - 1
    
    for i in range(N):
        row = x_new[i, :]
        softmax = np.exp(strength * row)
        softmax = softmax / np.sum(softmax)
        x_new[i, :] = 2 * softmax - 1
    
    return x_new.flatten()

def calculate_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    n = len(coordinates)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coordinates[i, 0] - coordinates[j, 0]
                dy = coordinates[i, 1] - coordinates[j, 1]
                distances[i, j] = np.sqrt(dx*dx + dy*dy)
    
    return distances

def extract_tsp_solution_with_legalizer(spin_config: np.ndarray, N: int, city_distances: np.ndarray = None) -> Tuple[List[int], bool, float]:
    spin_matrix = spin_config.reshape(N, N)
    path = [-1] * N
    
    # 统计每行每列的+1数量
    row_sums = np.sum(spin_matrix > 0, axis=1)  # 每个城市被访问次数
    col_sums = np.sum(spin_matrix > 0, axis=0)  # 每个位置的访问城市数
    
    # 检查原始解是否合法
    is_valid_original = True
    overlap_positions = []
    missing_positions = []
    duplicate_cities = []
    
    # 检查重叠（一个位置多个城市）
    for j in range(N):
        if col_sums[j] > 1:
            is_valid_original = False
            overlap_positions.append(j)
        elif col_sums[j] == 0:
            is_valid_original = False
            missing_positions.append(j)
    
    # 检查重复访问（一个城市访问多次）
    for i in range(N):
        if row_sums[i] > 1:
            is_valid_original = False
            duplicate_cities.append(i)
        elif row_sums[i] == 0:
            is_valid_original = False
    
    if is_valid_original:
        # 原始解合法，直接提取
        for j in range(N):
            for i in range(N):
                if spin_matrix[i, j] > 0:
                    path[j] = i
                    break
        cost = calculate_path_distance(path, city_distances) if city_distances is not None else 0
        return path, True, cost
    else:
        # 需要legalizer
        print(f"原始解不合法: 重叠位置{overlap_positions}, 缺失位置{missing_positions}, 重复城市{duplicate_cities}")
        legal_path, legal_cost = legalize_tsp_solution(spin_matrix, city_distances, N)
        return legal_path, False, legal_cost

def legalize_tsp_solution(spin_matrix: np.ndarray, city_distances: np.ndarray, N: int) -> Tuple[List[int], float]:
    """
    将不合法的TSP解修复为合法解
    
    策略：
    1. 首先处理重叠位置（多个城市争抢同一个位置）
    2. 然后处理缺失的城市（未被访问的城市）
    3. 使用贪心+局部搜索找到最优修复
    """
    # 获取所有+1的自旋位置
    positive_spins = []
    for i in range(N):
        for j in range(N):
            if spin_matrix[i, j] > 0:
                positive_spins.append((i, j, spin_matrix[i, j]))  # (城市, 位置, 自旋值)
    
    # 按自旋值排序（值越大表示置信度越高）
    positive_spins.sort(key=lambda x: x[2], reverse=True)
    
    # 贪心分配
    assigned_positions = set()
    assigned_cities = set()
    path = [-1] * N
    
    # 第一轮：分配高置信度的不冲突自旋
    for city, pos, confidence in positive_spins:
        if pos not in assigned_positions and city not in assigned_cities:
            path[pos] = city
            assigned_positions.add(pos)
            assigned_cities.add(city)
    
    # 第二轮：处理剩余冲突（重叠位置）
    for city, pos, confidence in positive_spins:
        if path[pos] == -1 and city not in assigned_cities:  # 位置空闲且城市未分配
            path[pos] = city
            assigned_positions.add(pos)
            assigned_cities.add(city)
    
    # 第三轮：处理缺失的城市和位置
    missing_cities = set(range(N)) - assigned_cities
    missing_positions = set(range(N)) - assigned_positions
    
    if missing_cities or missing_positions:
        print(f"需要修复: 缺失城市{len(missing_cities)}, 缺失位置{len(missing_positions)}")
        path = complete_missing_assignments(path, missing_cities, missing_positions, city_distances)
    
    # 验证修复后的解
    if not validate_tsp_path(path):
        print("修复失败，使用贪心构造")
        path = greedy_construct_from_spins(spin_matrix, city_distances, N)
    
    cost = calculate_path_distance(path, city_distances) if city_distances is not None else 0
    return path, cost

def complete_missing_assignments(path: List[int], missing_cities: set, missing_positions: set, 
                               city_distances: np.ndarray) -> List[int]:
    """完成缺失的分配"""
    if len(missing_cities) != len(missing_positions):
        print("错误：缺失城市和位置数量不匹配")
        # 强制匹配数量
        if len(missing_cities) > len(missing_positions):
            # 随机选择一些城市不访问（这不合理，应该重新分配）
            missing_cities = set(list(missing_cities)[:len(missing_positions)])
        else:
            missing_positions = set(list(missing_positions)[:len(missing_cities)])
    
    # 将缺失的城市分配到缺失的位置，最小化距离成本
    missing_cities_list = list(missing_cities)
    missing_positions_list = list(missing_positions)
    
    if not missing_cities_list:
        return path
    
    # 使用简单贪心：按位置顺序分配剩余城市
    for pos in missing_positions_list:
        if missing_cities_list:
            city = missing_cities_list.pop(0)
            path[pos] = city
    
    return path

def greedy_construct_from_spins(spin_matrix: np.ndarray, city_distances: np.ndarray, N: int) -> List[int]:
    """基于自旋矩阵贪心构造合法解"""
    # 计算每个城市-位置对的得分
    scores = spin_matrix.copy()
    
    path = [-1] * N
    used_cities = set()
    used_positions = set()
    
    # 多轮分配直到所有位置填满
    while len(used_positions) < N:
        # 找到当前最高得分的合法分配
        best_score = -np.inf
        best_city = -1
        best_pos = -1
        
        for i in range(N):
            if i in used_cities:
                continue
            for j in range(N):
                if j in used_positions:
                    continue
                if scores[i, j] > best_score:
                    best_score = scores[i, j]
                    best_city = i
                    best_pos = j
        
        if best_city != -1 and best_pos != -1:
            path[best_pos] = best_city
            used_cities.add(best_city)
            used_positions.add(best_pos)
        else:
            # 处理剩余分配
            remaining_cities = set(range(N)) - used_cities
            remaining_positions = set(range(N)) - used_positions
            for pos in remaining_positions:
                if remaining_cities:
                    city = remaining_cities.pop()
                    path[pos] = city
            break
    
    return path

def validate_tsp_path(path: List[int]) -> bool:
    """验证TSP路径是否合法"""
    if -1 in path:
        return False
    
    if len(set(path)) != len(path):
        return False
    
    if min(path) < 0 or max(path) >= len(path):
        return False
    
    return True

def calculate_path_cost_with_penalty(path: List[int], city_distances: np.ndarray, penalty_weight=1000) -> float:
    """计算路径成本，包含对不合法解的惩罚"""
    if not validate_tsp_path(path):
        # 计算非法性程度作为惩罚
        penalty = 0
        city_counts = {}
        for city in path:
            if city == -1:
                penalty += 1
            else:
                city_counts[city] = city_counts.get(city, 0) + 1
        
        for count in city_counts.values():
            if count > 1:
                penalty += (count - 1)
        
        return float('inf') if penalty > 0 else calculate_path_distance(path, city_distances) + penalty * penalty_weight
    else:
        return calculate_path_distance(path, city_distances)
    
problems = ["berlin52", "eil51", "st70"]
problem_dir = 'data/tsp/'
trials_num = 20

print(f'{"Problem":<12} {"Trial":<6} {"Valid":<6} {"Miss":<5} {"Overlap":<8} {"Distance":<10}')
print('-' * 55)

for problem in problems:
    for trial in range(trials_num):
        max_iterations=2000
        dimension, coordinates, name = read_tsplib_data(f"{problem_dir + problem}.tsp")

        # print(coordinates)
        distances = calculate_distance_matrix(coordinates)
        set_current_distance_matrix(distances)

        # print(distances)

        fixed_start = 0 if dimension > 50 else None

        J = tsp_to_hamiltonian(distances, fixed_start_city=fixed_start)

        energies, spins_config = bsb_tsp(
            J, num_iters=max_iterations, dt=0.05, constraint_strength=3.0
        )

        is_valid_tsp, overlap_count, missing_count = is_valid_tsp_solution(spins_config, dimension)
        final_path = extract_tsp_solution(spins_config, dimension)
        final_distance = calculate_path_distance(final_path, distances)
        
        print(f"{problem:<12} {trial:<6} {is_valid_tsp:<6} {missing_count:<5} {overlap_count:<8} {final_distance:>9.2f}")

        # valid_lengths = [l for l in path_lengths if l < float('inf')]
        # if valid_lengths:
        #     best_found = min(valid_lengths)
        #     print(f"best length: {best_found:.2f}")
