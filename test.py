#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import copy
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import time
import reprlib

from gnnLyx import gnnLyx
import myClass

################### 初始化参数 ###################

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache() #清除内存
torch.cuda.device_count()
print(f"Using device: {device}")

# 定义保存模型的目录
save_dir_models = "./models/"
save_dir_output = "./output/"

# gnn参数
hparams = {
    'link_state_dim': 32,   # link向量维数
    'path_state_dim': 32,   # path向量维数
    'T': 4,                 # 消息聚合次数
    'readout_units': 16,    # 读出层隐层维数
    'learn_embedding': True,    # If false, only the readout is trained
    'head_num': 4           # 多头注意力，头数，需整除状态向量的维数
}

# 训练参数
memory = deque(maxlen=20000)
batch_size = 64
eval_flag = False       # True: 推理模式,   False: 训练模式

################### 函数代码 ###################
def make_exprience(graph, fail_links, env_actions, reward):
    eval_link_attr, eval_path_attr, eval_mask = graph.get_features_one(fail_links, env_actions)
    return eval_link_attr, eval_path_attr, eval_mask, reward

def print_current_time(s:str):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{s} : {current_time}:\n")

def test_maxp(graph : myClass.m_graph, env_actions, fail_links):
    fail_flows = graph.get_fail_flows(env_actions, fail_links)
    print(f"fail_flows_id: {fail_flows}")
    print(f"fail_flows_path_link_id:")
    for flow_id in fail_flows:
        path = graph.flows[flow_id].paths[env_actions[flow_id]]
        print(f"flow {flow_id}: ")
        for i in range(len(path) - 1):
            link_id = graph.get_edgeId_by_node(path[i], path[i + 1])
            print(link_id, end=", ")
        print()

    max_p_list = graph.get_link_attr_max_fail_p_list(env_actions, fail_flows)
    indexed = [(value, idx) for idx, value in enumerate(max_p_list)]
    sorted_list = sorted(indexed, key=lambda x: x[0], reverse=True)
    print(f"fail_links: {fail_links}")
    for value, idx in sorted_list:
        print(f"idx: {idx},\tvalue: {value}")
    return

def calc_reward(graph, env_actions, fail_links, total_bw):
    fail_flows = graph.get_fail_flows(env_actions, fail_links)
    throughput = sum(graph.get_path_throughput(env_actions, fail_flows))
    return throughput / total_bw

def calc_performance(graph, env_actions, fail_links, total_bw, origin_throughput):
    fail_flows = graph.get_fail_flows(env_actions, fail_links)
    new_throughput = sum(graph.get_path_throughput(env_actions, fail_flows))
    performance = (new_throughput - origin_throughput) / (total_bw - origin_throughput + 1e-6)   # 防止除0
    print(f"performance\t = {performance:.4f}\t = ({new_throughput:.2f} - {origin_throughput:.2f}) / ({total_bw:.2f} - {origin_throughput:.2f})")
    return performance

################### 正式流程代码 ###################
try:
    print_current_time("start time")

    # 初始化覆盖txt
    with open(save_dir_output + "losses.txt", "w") as f:
        f.write(f"")
    with open(save_dir_output + "rewards.txt", "w") as f:
        f.write(f"")
    with open(save_dir_output + "performance.txt", "w") as f:
        f.write(f"")
    with open(save_dir_output + "best_performance.txt", "w") as f:
        f.write(f"")

    # 读入拓扑
    graph = myClass.m_graph()
    init_env_actions = graph.initial_generate_ba(100, 2)
    total_bw = sum([graph.flows[flow_id].bw for flow_id in range(graph.f)])
    print(f"n = {graph.n}, m = {graph.m}, f = {graph.f}")
    print(f"init_env_actions: \n{reprlib.repr(init_env_actions)}")

    # 初始化模型
    model = gnnLyx(hparams).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)    # 这里设置学习率
    if (eval_flag):
        model_state = torch.load("model_epoch_300000.pth", map_location=device)
        model.load_state_dict(model_state)

    print(f"model_device: {next(model.parameters()).device}") # 输出：cpu 或 cuda:0

    # 初始化变量
    n, m, flow_cnt = graph.n, graph.m, graph.f
    k = myClass.m_graph.K_SP_CNT

    # 初始化参数
    episodes = 1000000     # 跑多少轮
    FAIL_LINK_CNT_MIN, FAIL_LINK_CNT_MAX = 1, 3
    FAIL_FLOW_CNT_MIN, FAIL_FLOW_CNT_MAX = 2, 5

    # 统计时间
    time_eval_sum = 0.0
    time_train_sum = 0.0

    # ################### 强化学习 ###################
    # 环境: [env_actions + fail_links] -> fail_flows
    # 动作: 改变env_actions
    for now_episode in range(episodes):
        # 每轮开始时初始化
        env_actions = copy.deepcopy(init_env_actions)
        fail_links = []
        fail_flows = []
        while True:
            fail_links_cnt = random.randint(FAIL_LINK_CNT_MIN, FAIL_LINK_CNT_MAX)
            fail_links = random.sample(range(graph.m), fail_links_cnt)
            fail_flows = graph.get_fail_flows(env_actions, fail_links)
            if FAIL_FLOW_CNT_MIN <= len(fail_flows) <= FAIL_FLOW_CNT_MAX:
                break
        origin_throughput = sum(graph.get_path_throughput(env_actions, fail_flows))
        
        print(f"\n=== Episode {now_episode} ===")

        # ################### 推理模式 ###################
        # 1、枚举所有方案
        # 2、计算奖励
        # 3、推理，比较q值和实际奖励
        # 4、经验入库
        start_eval_time = time.perf_counter()
        link_attr_batch, path_attr_batch, mask_batch, new_actions_list = graph.get_features(env_actions, fail_flows, device)
        rewards = []
        for acts in new_actions_list:
            r = calc_reward(graph, acts, fail_links, total_bw)
            rewards.append(r)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        best_idx = torch.argmax(rewards_tensor).item()

        best_env_actions = new_actions_list[best_idx]
        best_reward = rewards[best_idx]
        best_performance = calc_performance(graph, best_env_actions, fail_links, total_bw, origin_throughput)

        q_idx = -1
        model.eval()
        with torch.no_grad():
            q_values = model(link_attr_batch, path_attr_batch, mask_batch)
            q_idx = torch.argmax(q_values).item()

        q_env_actions = new_actions_list[q_idx]
        q_reward = rewards[q_idx]
        q_performance = calc_performance(graph, q_env_actions, fail_links, total_bw, origin_throughput)

        print(f"Best reward:\t idx = {best_idx}, reward = {best_reward:.4f}, performance = {best_performance:.4f}")
        print(f"Q    reward:\t idx = {q_idx}, reward = {q_reward:.4f}, performance = {q_performance:.4f}")
        with open(save_dir_output + "rewards.txt", "a") as f:
            f.write(f"{now_episode},{q_reward}\n")
        with open(save_dir_output + "performance.txt", "a") as f:
            f.write(f"{now_episode},{q_performance}\n")
        with open(save_dir_output + "best_performance.txt", "a") as f:
            f.write(f"{now_episode},{best_performance}\n")

        # 经验入库
        experience = make_exprience(graph, fail_links, best_env_actions, best_reward)
        memory.append(copy.deepcopy(experience))
        experience_q = make_exprience(graph, fail_links, q_env_actions, q_reward)
        memory.append(copy.deepcopy(experience_q))

        time_eval = (time.perf_counter() - start_eval_time) * 1000
        time_eval_sum += time_eval

        # ################### 训练模式 ###################
        time_train_start = time.perf_counter()
        if len(memory) > batch_size:
            batch_samples = random.sample(memory, batch_size)
            
            batch = random.sample(memory, batch_size)
            eval_link_attr, eval_path_attr, eval_mask, exp_rewards = zip(*batch)

            # 训练模式
            model.train()
            eval_q_values = model(torch.tensor(eval_link_attr, device=device), torch.tensor(eval_path_attr, device=device), torch.tensor(eval_mask, device=device))
            target_q_values = torch.tensor(exp_rewards, device=device)
            loss = nn.functional.mse_loss(eval_q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with open(save_dir_output + "losses.txt", "a") as f:
                f.write(f"{now_episode},{loss.item()}\n")
        time_train = (time.perf_counter() - time_train_start) * 1000
        time_train_sum += time_train
        print(f"eval : {time_eval:.3f} ms")
        print(f"train : {time_train:.3f} ms")

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    print_current_time("end time")
    print("Training finished.")
    print(f"Total episodes: {now_episode + 1}")
    print(f"Avg eval : {time_eval_sum / (now_episode + 1):.3f} ms")
    print(f"Avg train : {time_train_sum / (now_episode + 1):.3f} ms")
    print("Done")