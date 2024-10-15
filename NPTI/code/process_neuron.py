import torch
import os
import json
import argparse
parser = argparse.ArgumentParser()
# 定义目录路径
parser = argparse.ArgumentParser()
parser.add_argument("--neuron_dir", type=str, required=True, help="Directory for saving neuron results")
args = parser.parse_args()
directory = args.neuron_dir
def read_and_convert_to_dict(file_path, num_lines, min_required=2000):
    result_dict = {}
    t = 0
    reached_minimum = False

    with open(file_path, 'r') as f:
        for idx in range(num_lines):
            line = f.readline().strip()
            if not line:
                break

            layer = eval(line)[0]
            difference = eval(line)[2]

            if idx < min_required or (idx >= min_required and difference >=0.1):
                if layer not in result_dict:
                    result_dict[layer] = []
                result_dict[layer].append(eval(line))
                t += 1

            if t >= min_required and difference <0.1:
                reached_minimum = True
                break

    if not reached_minimum:
        print(f"Warning: Not enough data with difference >= 0.1, total added: {t}")

    print(t, file_path)
    return result_dict

def save_dict_to_json(dict_obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(dict_obj, f, indent=4)
# 处理和保存差异
def process_and_save_differences(tensor1, tensor2, num_cols, save_path):
    differences = tensor1.view(-1) - tensor2.view(-1)
    sorted_differences, sorted_order = torch.sort(differences, descending=True)
    sorted_indices = sorted_order
    sorted_layer_positions = sorted_indices // num_cols
    sorted_col_positions = sorted_indices % num_cols
    corresponding_values = tensor1.view(-1)[sorted_indices]

    sorted_tuples = list(zip(
        sorted_layer_positions.tolist(),
        sorted_col_positions.tolist(),
        sorted_differences.tolist(),
        corresponding_values.tolist()
    ))

    # 将结果保存到文件，每行一个元组
    with open(save_path, 'w') as f:
        for item in sorted_tuples:
            f.write(f"{item}\n")

# 计算直方图的分位数
def calculate_quantiles(histogram, bins, quantiles=[0.95]):
    cumsum = torch.cumsum(histogram, dim=-1).float()
    total = cumsum[-1]
    results = []

    for q in quantiles:
        threshold = q * total
        index = torch.searchsorted(cumsum, threshold)
        results.append(bins[index.item()].item())

    return results

# 定义所有处理类型
types = [
    'Neuroticism', 'Openness', 
    'Conscientiousness', 'Extraversion', 
    'Agreeableness'
]

# 处理每个类型的文件
for bfi in types:
    # 加载over_zero数据
    file = f'{directory}/{bfi}.pt'
    file_reversed = f'{directory}/{bfi}_reversed.pt'
    data = torch.load(file)
    data_reversed = torch.load(file_reversed)
    
    over_zero_prob = data['over_zero'] / data['token_num']
    over_zero_reversed_prob = data_reversed['over_zero'] / data_reversed['token_num']
    _, num_cols = over_zero_prob.shape

    # 处理并保存差值
    save_path = f'{directory}/{bfi}_sorted_differences.txt'
    process_and_save_differences(over_zero_prob, over_zero_reversed_prob, num_cols, save_path)
    
    save_path_reversed = f'{directory}/{bfi}_reversed_sorted_differences.txt'
    process_and_save_differences(over_zero_reversed_prob, over_zero_prob, num_cols, save_path_reversed)

    print(f"{bfi} processed, results saved to {save_path} and {save_path_reversed}.")
    for mode in ["","_reversed"]:
        # 处理直方图
        t=bfi+mode
        histograms_path = f'{directory}/{t}.pt'
        histograms_dict = torch.load(histograms_path)
        histograms = histograms_dict['histograms']

        # 定义直方图的bins
        bins = torch.cat([torch.tensor([-float('inf')]), torch.arange(0, 3.01, 0.01)])
        bins[-1] = float('inf')

        # 加载排序后的差异元组
        tuples_path = f'{directory}/{t}_sorted_differences.txt'
        with open(tuples_path, 'r') as file:
            tuples = []
            for line in file:
                parts = line.strip().strip('()').split(', ')
                i = int(parts[0])
                j = int(parts[1])
                k = float(parts[2])
                l = float(parts[3])
                tuples.append((i, j, k, l))

        # 处理每个元组，计算分位数
        updated_tuples = []
        for i, j, k, l in tuples:
            histogram = histograms[i][j].cpu()  # 将直方图移至CPU进行处理
            quantiles = calculate_quantiles(histogram, bins)
            updated_tuple = (i, j, k, l) + tuple(quantiles)
            updated_tuples.append(updated_tuple)

        # 保存更新后的元组到文件
        with open(tuples_path, 'w') as file:
            for tup in updated_tuples:
                file.write(f"{tup}\n")

        num_lines = 300000
        save_path = f'{directory}/{t}_sorted_differences.txt'
        save_path_dict = f'{directory}/{t}_dict.json'
        _dict = read_and_convert_to_dict(save_path, num_lines)
        save_dict_to_json(_dict, save_path_dict)
###################################转换成字典################################################
# import json
# from IPython import embed
# def read_and_convert_to_dict(file_path, num_lines, min_required=2000):
#     result_dict = {}
#     t = 0
#     reached_minimum = False

#     with open(file_path, 'r') as f:
#         for idx in range(num_lines):
#             line = f.readline().strip()
#             if not line:
#                 break

#             layer = eval(line)[0]
#             difference = eval(line)[2]

#             if idx < min_required or (idx >= min_required and difference >=0.1):
#                 if layer not in result_dict:
#                     result_dict[layer] = []
#                 result_dict[layer].append(eval(line))
#                 t += 1

#             if t >= min_required and difference <0.1:
#                 reached_minimum = True
#                 break

#     if not reached_minimum:
#         print(f"Warning: Not enough data with difference >= 0.1, total added: {t}")

#     print(t, file_path)
#     return result_dict


# def save_dict_to_json(dict_obj, file_path):
#     with open(file_path, 'w') as f:
#         json.dump(dict_obj, f, indent=4)

# directory="/home/dengjia/roleplay12_phi/neuron_results"
# for t in['Neuroticism','Agreeableness','Extraversion','Conscientiousness','Openness']:
#     num_lines = 300000
#     save_path = f'{directory}/{t}_sorted_differences.txt'
#     save_path_reversed = f'{directory}/{t}_reversed_sorted_differences.txt'
#     save_path_dict = f'{directory}/{t}_dict.json'
#     save_path_reversed_dict = f'{{directory}}/{t}_reversed_dict.json'

#     _dict = read_and_convert_to_dict(save_path, num_lines)
#     reversed_dict = read_and_convert_to_dict(save_path_reversed, num_lines)
#     save_dict_to_json(_dict, save_path_dict)
#     save_dict_to_json(reversed_dict, save_path_reversed_dict)