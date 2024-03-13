import pickle

# 读取 pickle 文件
with open('/root/deepnetwork_backdoor_attack_original/BackdoorBench/record/badnet_0_1/train_poison_index_list.pickle', 'rb') as f:
    data = pickle.load(f)

# 打印 pickle 文件中的内容
print(data)