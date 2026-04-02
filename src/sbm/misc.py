import numpy as np
import pandas as pd
import os
import time
from scipy.sparse import random as sparse_random
from scipy.sparse import coo_matrix
from collections import deque

def load_data(name='data/Gset/G30'):
    file = open(name, 'r')
    for (idx, line) in enumerate(file):
        if idx == 0:
            N = int(line.split(' ')[0])
            J = np.zeros([N,N])
        else:
            J[int(line.split(' ')[0])-1][int(line.split(' ')[1])-1] = (line.split(' ')[2])
    file.close()
    tor_arr = -J
    return tor_arr

def load_qplib_data(file_path):
    """
    Revised QPLIB loader based on official documentation:
    Objective: 1/2 x^T Q^0 x + b^0 x + q^0
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    num_vars = int(lines[3].split('#')[0].strip())
    objective_sense = lines[2].lower() # minimize or maximize
    
    Q = np.zeros((num_vars, num_vars))
    b = np.zeros(num_vars)
    
    for idx, line in enumerate(lines):
        if 'number of quadratic terms in objective' in line:
            num_terms = int(line.split('#')[0][0:idx].strip()) if '#' in line else int(line.split()[0])
            # The line itself may contain the count, but we've already split it. 
            # Re-parsing with split logic for robustness
            num_terms = int(line.split()[0])

            for i in range(1, num_terms + 1):
                parts = lines[idx + i].split()
                v1, v2, val = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
                Q[v1, v2] = val
                # print(f"Parsed quadratic term: Q[{v1}, {v2}] = {val}")
        elif 'default value for linear coefficients in objective' in line:
            default_b = float(line.split()[0])
            b.fill(default_b)
        elif 'number of non-default linear coefficients in objective' in line:
            num_terms = int(line.split()[0])
            for i in range(1, num_terms + 1):
                parts = lines[idx + i].split()
                v1, val = int(parts[0]) - 1, float(parts[1])
                b[v1] = val
                # print(f"Parsed linear term: b[{v1}] = {val}")
        
    return Q, b, num_vars, objective_sense

def export_configs_to_csv(csv_data_all, filename='config_energy_results.csv'):
    data_rows = []
    
    for graph_name, graph_data in csv_data_all.items():
        config_a_row = {'Config': 'Config_A_bsb', 'Graph': graph_name}
        
        for batch_idx in range(100):  # batch_size = 100
            key = f'bsb_batch_{batch_idx}'
            if key in graph_data:
                config_a_row[f'batch_{batch_idx}'] = graph_data[key]

        config_b_row = {'Config': 'Config_B_qsb_no_tuned', 'Graph': graph_name}
    
        for batch_idx in range(100):
            key = f'qsb_no_tuned_batch_{batch_idx}'
            if key in graph_data:
                config_b_row[f'batch_{batch_idx}'] = graph_data[key]
        
        data_rows.append(config_a_row)
        data_rows.append(config_b_row)
    
    df = pd.DataFrame(data_rows)
    df.to_csv(filename, index=False)
    return df

def append_config_c_to_csv(graph_name, qsb_finals, filename='config_energy_results.csv'):
    config_c_row = {'Config': 'Config_C_qsb_tuned', 'Graph': graph_name}
    
    for batch_idx, value in enumerate(qsb_finals):
        config_c_row[f'batch_{batch_idx}'] = value
    
    new_data_df = pd.DataFrame([config_c_row])

    if not os.path.exists(filename):
        new_data_df.to_csv(filename, index=False)
        print(f"Config C data for {graph_name} saved to {filename}")
    else:
        existing_df = pd.read_csv(filename)
        existing_row_idx = existing_df[(existing_df['Config'] == 'Config_C_qsb_tuned') & 
                                      (existing_df['Graph'] == graph_name)].index
        
        if len(existing_row_idx) > 0:
            existing_df.loc[existing_row_idx[0]] = config_c_row
            updated_df = existing_df
            print(f"Config C data for {graph_name} updated in {filename}")
        else:
            updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)
            print(f"Config C data for {graph_name} appended to {filename}")
        
        updated_df.to_csv(filename, index=False)

def tileElementCount(J, tileWidth = 64):
    N = J.shape[0]

    # resize original matrix
    tileNum = (N + tileWidth - 1) // tileWidth  # 向上取整
    newSize = tileNum * tileWidth
    newJ = np.resize(J, (newSize, newSize))  

    tileInfo = []

    for i in range(tileNum):
        for j in range(tileNum):
            tile = newJ[i * tileWidth: (i + 1) * tileWidth, j * tileWidth: (j + 1) * tileWidth]
            nnzCount = np.count_nonzero(tile)
            tileInfo.append((tile, nnzCount))

    # for idx, (tile, count) in enumerate(tileInfo):
    #     print(f"Submatrix {idx + 1}:\nElement Count: {count}\n")

    return tileInfo, tileNum * tileNum

def cbElementCount(J, tileWidth = 64, K = 8):
    N = J.shape[0]

    # resize original matrix
    cbWidth = K * tileWidth
    newSize = (N // cbWidth + 1) * cbWidth
    newJ = np.resize(J, (newSize, newSize))  

    tileX = (N + tileWidth - 1) // tileWidth  # 向上取整
    cbY = (N + cbWidth - 1) // cbWidth        # 向上取整

    cbInfo = []

    for j in range(cbY):
        cbInfo.append([])
        for i in range(tileX):
            cb = newJ[i * tileWidth: (i + 1) * tileWidth, j * cbWidth: (j + 1) * cbWidth]
            nnzCount = np.count_nonzero(cb)
            cbInfo[j].append(nnzCount + 3)

    return cbInfo

def binPack(cbInfo = [[]], peNum = 4):
    
    peWorkload = [0, 0, 0, 0]
    maxWorkloadDiff = 0
    minWorkloadDiff = 9999999
    AvyWorkloadDiff = 0
    efficiency = 0
    totalCycle = 0

    for rb in cbInfo:
        sortedRb = rb
        sortedRb.sort(reverse=True) 
        while (len(sortedRb) != 0):
            cbLength = sortedRb.pop(0)
            minWorkload = min(peWorkload)
            minWorkloadIdx = peWorkload.index(minWorkload)
            peWorkload[minWorkloadIdx] += cbLength
        maxWorkload = max(peWorkload)

        totalCycle += maxWorkload / 8

        for workload in peWorkload:
            efficiency += workload / maxWorkload

        WorkloadDiff = maxWorkload - min(peWorkload)
        maxWorkloadDiff = max(maxWorkloadDiff, WorkloadDiff)
        minWorkloadDiff = min(minWorkloadDiff, WorkloadDiff)
        AvyWorkloadDiff += WorkloadDiff

    AvyWorkloadDiff /= len(cbInfo)
    efficiency /= peNum * len(cbInfo)

    totalRunTime = totalCycle * 3.33 / 1000 # in ms

    return (totalRunTime, AvyWorkloadDiff, maxWorkloadDiff, minWorkloadDiff, efficiency)

def normalPack(cbInfo = [[]], peNum = 4):
    peWorkload = [0] * peNum
    maxWorkloadDiff = 0
    minWorkloadDiff = 9999999
    AvyWorkloadDiff = 0
    efficiency = 0
    totalCycle = 0

    for rb in cbInfo:
        current_pe = 0
        for cbLength in rb:
            peWorkload[current_pe] += cbLength
            current_pe = (current_pe + 1) % peNum
        
        maxWorkload = max(peWorkload)
        minWorkload = min(peWorkload)

        for workload in peWorkload:
            efficiency += workload / maxWorkload

        totalCycle += maxWorkload / 8

        WorkloadDiff = maxWorkload - minWorkload
        maxWorkloadDiff = max(maxWorkloadDiff, WorkloadDiff)
        minWorkloadDiff = min(minWorkloadDiff, WorkloadDiff)
        AvyWorkloadDiff += WorkloadDiff

    AvyWorkloadDiff /= len(cbInfo)
    efficiency /= peNum * len(cbInfo)

    totalRunTime = totalCycle * 3.33  / 1000 # in ms

    return (totalRunTime, AvyWorkloadDiff, maxWorkloadDiff, minWorkloadDiff, efficiency)

def sparsityAnalysis(fileStart=30, fileEnd=30):
    
    for idx in range(fileStart, fileEnd):
        name = './data/Gset/G' + str(idx)
        file = open(name, 'r')
        line = file.readline().split(' ')
        N = int(line[0])
        nnzCount = int(line[1])
        sparsity = nnzCount / (N**2)
        print("File name: ", name, ", size ", N, f", sparsity: {sparsity:.4f}")

def genSparseMatrix(matrixSize = 4000, startIdx = 1):
    sparsityList = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]

    idx = startIdx
    for sparsity in sparsityList:
        num_nonzero_elements = int((1 - sparsity) * matrixSize * matrixSize)
        sparse_matrix = sparse_random(matrixSize, matrixSize, density=1 - sparsity, format='coo', random_state=42, dtype=int)
        coo = coo_matrix(sparse_matrix)
        with open('data/test/S' + str(idx), 'w') as f:
            f.write(f"{matrixSize} {sparse_matrix.nnz}\n")
            for i in range(coo.nnz):
                f.write(f"{coo.row[i]} {coo.col[i]} {1}\n")

        idx += 1

def tileElementInfo(name='Gset/G34', tileWidth=64):
    J = load_data(name)
    N = J.shape[0]

    # resize original matrix
    tileNum = (N + tileWidth - 1) // tileWidth  # 向上取整
    newSize = tileNum * tileWidth
    newJ = np.resize(J, (newSize, newSize))  

    tileInfo = []

    for i in range(tileNum):
        for j in range(tileNum):
            tile = newJ[i * tileWidth: (i + 1) * tileWidth, j * tileWidth: (j + 1) * tileWidth]
            
            # 获取真实的非零元素位置和值
            nnz_positions = []
            nnz_values = []
            
            # 在tile内找到所有非零元素
            for local_row in range(tileWidth):
                for local_col in range(tileWidth):
                    global_row = i * tileWidth + local_row
                    global_col = j * tileWidth + local_col
                    value = tile[local_row, local_col]
                    
                    if value != 0:
                        nnz_positions.append({
                            'global_row': global_row,
                            'global_col': global_col,
                            'local_row': local_row,
                            'local_col': local_col,
                            'tile_row': i,
                            'tile_col': j
                        })
                        nnz_values.append(value)
            
            nnzCount = len(nnz_positions)
            tileInfo.append({
                'tile_data': tile,
                'nnz_count': nnzCount,
                'nnz_positions': nnz_positions,
                'nnz_values': nnz_values,
                'tile_coord': (i, j)
            })
    return tileInfo

def cbElementInfo(name='Gset/G34', tileWidth=64, K=8):
    J = load_data(name)
    N = J.shape[0]

    # resize original matrix
    cbWidth = K * tileWidth
    newSize = (N // cbWidth + 1) * cbWidth
    newJ = np.resize(J, (newSize, newSize))  

    tileX = (N + tileWidth - 1) // tileWidth  
    cbY = (N + cbWidth - 1) // cbWidth       

    cbInfo = []

    for j in range(cbY):
        cb_row_info = []
        for i in range(tileX):
            # 提取CB区域
            cb_region = newJ[i * tileWidth: (i + 1) * tileWidth, j * cbWidth: (j + 1) * cbWidth]
            
            # 收集CB内所有非零元素的详细信息
            cb_nnz_details = []
            
            for local_tile_row in range(tileWidth):
                for local_tile_col in range(cbWidth):  # CB宽度是K * tileWidth
                    global_row = i * tileWidth + local_tile_row
                    global_col = j * cbWidth + local_tile_col
                    value = cb_region[local_tile_row, local_tile_col]
                    
                    if value != 0:
                        # 计算在CB内的tile索引和局部位置
                        tile_idx_in_cb = local_tile_col // tileWidth
                        local_col_in_tile = local_tile_col % tileWidth
                        
                        cb_nnz_details.append({
                            'global_row': global_row,
                            'global_col': global_col,
                            'local_row_in_tile': local_tile_row,  # 在tile内的行
                            'local_col_in_tile': local_col_in_tile,  # 在tile内的列
                            'tile_idx_in_cb': tile_idx_in_cb,  # 在CB中的tile索引
                            'value': value,
                            'cb_coord': (i, j),
                            'tile_coord_in_cb': (i, tile_idx_in_cb)
                        })
            
            nnzCount = len(cb_nnz_details)
            cb_row_info.append({
                'nnz_count': nnzCount + 3,  # 保持原来的+3逻辑
                'nnz_details': cb_nnz_details,
                'cb_region': cb_region,
                'tile_coord': (i, j)
            })
        
        cbInfo.append(cb_row_info)

    return cbInfo

def reorderHazardElements(cbInfo, tileWidth=64, max_attempts=3):
    start_time = time.time()
    
    total_original_elements = 0
    total_final_elements = 0
    total_padding_added = 0
    reordered_sequences = []
    
    for cb_idx, cb_row in enumerate(cbInfo):
        element_sequences = []
        
        for tile_info in cb_row:
            tile_elements = []
            nnz_details = tile_info['nnz_details']
            
            for element in nnz_details:
                detailed_element = {
                    'tile_idx': element['tile_idx_in_cb'],
                    'global_row': element['global_row'],
                    'global_col': element['global_col'],
                    'local_row': element['local_row_in_tile'],  
                    'local_col': element['local_col_in_tile'], 
                    'value': element['value'],
                    'cb_coord': element['cb_coord'],
                    'is_nnz': True
                }
                tile_elements.append(detailed_element)
            
            element_sequences.append(tile_elements)
            total_original_elements += len(tile_elements)
        
        reordered_cb = []
        row_buffers = [deque() for _ in range(tileWidth)]
        
        for tile_idx, tile_elements in enumerate(element_sequences):
            for element in tile_elements:
                row_buffers[element['local_row']].append(element)
        
        attempt_successful = False
        for attempt in range(max_attempts):
            temp_buffers = [deque(buffer) for buffer in row_buffers]
            temp_reordered = []
            cycle_conflicts = 0
            
            while any(temp_buffers):
                cycle_elements = []
                used_rows = set()
                
                for row in range(tileWidth):
                    if temp_buffers[row] and row not in used_rows:
                        element = temp_buffers[row].popleft()
                        cycle_elements.append(element)
                        used_rows.add(row)
                
                remaining_conflicts = sum(1 for row in range(tileWidth) 
                                        if temp_buffers[row] and row not in used_rows)
                
                if remaining_conflicts > 0:
                    cycle_conflicts += remaining_conflicts
                
                temp_reordered.append(cycle_elements)
            
            if cycle_conflicts == 0:
                reordered_cb = temp_reordered
                attempt_successful = True
                break
        
        if not attempt_successful:
            max_cycle_length = 0
            for cycle in temp_reordered:
                max_cycle_length = max(max_cycle_length, len(cycle))
            
            padded_reordered = []
            for cycle in temp_reordered:
                padded_cycle = cycle.copy()
                while len(padded_cycle) < max_cycle_length:
                    padding_element = {
                        'tile_idx': -1,
                        'global_row': -1,
                        'global_col': -1,
                        'local_row': -1,
                        'local_col': -1,
                        'value': 0,
                        'cb_coord': (-1, -1),
                        'is_nnz': False
                    }
                    padded_cycle.append(padding_element)
                    total_padding_added += 1
                padded_reordered.append(padded_cycle)
            
            reordered_cb = padded_reordered
        
        for cycle in reordered_cb:
            total_final_elements += len(cycle)
        
        reordered_sequences.append(reordered_cb)
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        'original_elements': total_original_elements,
        'final_elements': total_final_elements,
        'processing_time': processing_time,
        'padding_added': total_padding_added,
        'overhead_ratio': total_final_elements / total_original_elements if total_original_elements > 0 else 1.0,
        'efficiency_ratio': total_original_elements / total_final_elements if total_final_elements > 0 else 0.0,
        'reordered_sequences': reordered_sequences,
        'strategy_used': 'reordering_only' if total_padding_added == 0 else 'reordering_with_padding'
    }

def test_skewed_sparsity_graphs():
    tile_widths = [64, 128, 256]
    K = 8
    
    test_graphs = []

    fixed_nnz = 500000 
    # n_values = [1000, 2000, 4000, 8000, 12000, 16000]
    n_values = [1000, 2000, 4000, 8000]

    file_dir = "Aset/"
    for n in n_values:

        test_graphs.append(file_dir + "random_" + str(n) + "_" + str(fixed_nnz) + ".hgr")
        test_graphs.append(file_dir + "powerlaw_" + str(n) + "_" + str(fixed_nnz) + ".hgr")
        test_graphs.append(file_dir + "skewedblock_" + str(n) + "_" + str(fixed_nnz) + ".hgr")
        test_graphs.append(file_dir + "community_" + str(n) + "_" + str(fixed_nnz) + ".hgr")
        test_graphs.append(file_dir + "scalefree_" + str(n) + "_" + str(fixed_nnz) + ".hgr")

    # fixed_n = 8000 
    # nnz_values = [10000, 40000, 80000, 160000, 320000, 640000]

    # for nnz in nnz_values:
    #     test_graphs.append(file_dir + "random_" + str(fixed_n) + "_" + str(nnz) + ".hgr")
    #     test_graphs.append(file_dir + "powerlaw_" + str(fixed_n) + "_" + str(nnz) + ".hgr")
    #     test_graphs.append(file_dir + "skewedblock_" + str(fixed_n) + "_" + str(nnz) + ".hgr")
    #     test_graphs.append(file_dir + "community_" + str(fixed_n) + "_" + str(nnz) + ".hgr")
    #     test_graphs.append(file_dir + "scalefree_" + str(fixed_n) + "_" + str(nnz) + ".hgr")
    
    results = {}

    print(f'graph,    efficiency')
    
    for graph_file in test_graphs:        
        efficiency_results = []
        runtime_results = []
        
        for tile_width in tile_widths:
            J = load_data(graph_file)
            cb_info = cbElementCount(J, tile_width, K)
            totalRunTime, _, _, _, efficiency = binPack(cb_info)
            
            efficiency_results.append(efficiency)
            runtime_results.append(totalRunTime)
        
        results[graph_file] = {
            'efficiency': efficiency_results,
            'runtime': runtime_results
        }

        print(f'{graph_file}, {efficiency_results}')
    
    return results, tile_widths
