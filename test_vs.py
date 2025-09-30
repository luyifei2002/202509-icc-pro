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

from gnnLyx import gnnLyx
import myClass

################### 初始化参数 ###################

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache() #清除内存
torch.cuda.device_count()
print(f"Using device: {device}")

# 定义保存模型的目录
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

# 记录list
rewards = []
losses = []

################### 函数代码 ###################
def print_current_time(s:str):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{s} time : {current_time}:\n")

def get_reward_model(graph:myClass.m_graph, _env_actions, fail_links):
    env_actions = copy.deepcopy(_env_actions)
    fail_flows = graph.get_fail_flows(env_actions, fail_links)
    origin_fail_flows_cnt = len(fail_flows)
    stepIdx = 0
    reward = 0
    total_reward = 0
    done = False
    cnt = 0
    while not done:
        stepIdx += 1
        last_fail_flows = fail_flows

        # ################### 推理模式 ###################
        model.eval()
        with torch.no_grad():
            link_attr, path_attr, mask, new_actions_list = graph.get_features(env_actions, fail_flows, device)
            q_values = model(link_attr, path_attr, mask)
            _, max_q_index = torch.max(q_values, dim=0)
            best_actions_index = max_q_index.item()
            env_actions = new_actions_list[best_actions_index]
            if stepIdx == 1:
                cnt = len(new_actions_list)
        
        fail_flows = graph.get_fail_flows(env_actions, fail_links)
        reward = 1.0 * (len(last_fail_flows) - len(fail_flows)) / origin_fail_flows_cnt / stepIdx
        total_reward += reward

        if len(fail_flows) == 0:                                # 成功重路由, 认为done
            done = True
        if fail_flows == last_fail_flows:                       # 做出动作没任何效果, 认为done, 且由于没做出改变, 所以直接continue
            done = True
    print(f"model_reward: \t{total_reward:.2f}", end=", \t")
    return total_reward, cnt

def get_reward_random(graph:myClass.m_graph, _env_actions, fail_links):
    env_actions = copy.deepcopy(_env_actions)
    fail_flows = graph.get_fail_flows(env_actions, fail_links)
    origin_fail_flows_cnt = len(fail_flows)
    stepIdx = 0
    reward = 0
    total_reward = 0
    done = False
    while not done:
        stepIdx += 1
        last_fail_flows = fail_flows

        _, _, _, new_actions_list = graph.get_features(env_actions, fail_flows, device)
        index_random = random.randint(0, len(new_actions_list) - 1)
        env_actions = new_actions_list[index_random]
        
        fail_flows = graph.get_fail_flows(env_actions, fail_links)
        reward = 1.0 * (len(last_fail_flows) - len(fail_flows)) / origin_fail_flows_cnt / stepIdx
        total_reward += reward

        if len(fail_flows) == 0:                                # 成功重路由, 认为done
            done = True
        if fail_flows == last_fail_flows:                       # 做出动作没任何效果, 认为done, 且由于没做出改变, 所以直接continue
            done = True
    print(f"random_reward: \t{total_reward:.2f}", end=", \t")
    return total_reward

def get_reward_greed_sum(graph:myClass.m_graph, _env_actions, fail_links):
    env_actions = copy.deepcopy(_env_actions)
    fail_flows = graph.get_fail_flows(env_actions, fail_links)
    origin_fail_flows_cnt = len(fail_flows)
    stepIdx = 0
    reward = 0
    total_reward = 0
    done = False
    while not done:
        stepIdx += 1
        last_fail_flows = fail_flows

        link_attr, _, mask, new_actions_list = graph.get_features(env_actions, fail_flows, device)
        greed_values = []
        for betch_id in range(len(new_actions_list)):
            greed_value = 0
            for flow_id in fail_flows:
                path = graph.flows[flow_id].paths[new_actions_list[betch_id][flow_id]]
                for j in range(len(path) - 1):
                    link_id = graph.get_edgeId_by_node(path[j], path[j + 1])
                    greed_value += link_attr[betch_id][link_id][3]      # 计算方案中fail_p的和，取和最小的方案来贪心
            greed_values.append(greed_value)
        best_actions_index = greed_values.index(min(greed_values))
        env_actions = new_actions_list[best_actions_index]
        
        fail_flows = graph.get_fail_flows(env_actions, fail_links)
        reward = 1.0 * (len(last_fail_flows) - len(fail_flows)) / origin_fail_flows_cnt / stepIdx
        total_reward += reward

        if len(fail_flows) == 0:                                # 成功重路由, 认为done
            done = True
        if fail_flows == last_fail_flows:                       # 做出动作没任何效果, 认为done, 且由于没做出改变, 所以直接continue
            done = True
    print(f"greed_sum_reward: \t{total_reward:.2f}", end=", \t")
    return total_reward

def get_reward_greed_min(graph:myClass.m_graph, _env_actions, fail_links):
    env_actions = copy.deepcopy(_env_actions)
    fail_flows = graph.get_fail_flows(env_actions, fail_links)
    origin_fail_flows_cnt = len(fail_flows)
    stepIdx = 0
    reward = 0
    total_reward = 0
    done = False
    while not done:
        stepIdx += 1
        last_fail_flows = fail_flows

        link_attr, _, mask, new_actions_list = graph.get_features(env_actions, fail_flows, device)
        greed_values = []
        for betch_id in range(len(new_actions_list)):
            greed_value = 0
            for flow_id in fail_flows:
                path = graph.flows[flow_id].paths[new_actions_list[betch_id][flow_id]]
                for j in range(len(path) - 1):
                    link_id = graph.get_edgeId_by_node(path[j], path[j + 1])
                    greed_value = max(greed_value, link_attr[betch_id][link_id][3])      # 计算方案中fail_p的最大值，取最大值的最大值最小的方案来贪心
            greed_values.append(greed_value)
        best_actions_index = greed_values.index(min(greed_values))
        env_actions = new_actions_list[best_actions_index]
        
        fail_flows = graph.get_fail_flows(env_actions, fail_links)
        reward = 1.0 * (len(last_fail_flows) - len(fail_flows)) / origin_fail_flows_cnt / stepIdx
        total_reward += reward

        if len(fail_flows) == 0:                                # 成功重路由, 认为done
            done = True
        if fail_flows == last_fail_flows:                       # 做出动作没任何效果, 认为done, 且由于没做出改变, 所以直接continue
            done = True
    print(f"greed_min_reward: \t{total_reward:.2f}", end=", \t")
    return total_reward

def get_reward_greed_max_sum(graph:myClass.m_graph, _env_actions, fail_links):
    env_actions = copy.deepcopy(_env_actions)
    fail_flows = graph.get_fail_flows(env_actions, fail_links)
    origin_fail_flows_cnt = len(fail_flows)
    stepIdx = 0
    reward = 0
    total_reward = 0
    done = False
    while not done:
        stepIdx += 1
        last_fail_flows = fail_flows

        link_attr, _, mask, new_actions_list = graph.get_features(env_actions, fail_flows, device)
        greed_values = []
        for betch_id in range(len(new_actions_list)):
            greed_value = 0
            for flow_id in fail_flows:
                path = graph.flows[flow_id].paths[new_actions_list[betch_id][flow_id]]
                max_value = 0
                for j in range(len(path) - 1):
                    link_id = graph.get_edgeId_by_node(path[j], path[j + 1])
                    max_value = max(max_value, link_attr[betch_id][link_id][3])      # 计算方案中fail_p的最大值，取最大值的和最小的方案来贪心
                greed_value += max_value
            greed_values.append(greed_value)
        best_actions_index = greed_values.index(min(greed_values))
        env_actions = new_actions_list[best_actions_index]
        
        fail_flows = graph.get_fail_flows(env_actions, fail_links)
        reward = 1.0 * (len(last_fail_flows) - len(fail_flows)) / origin_fail_flows_cnt / stepIdx
        total_reward += reward

        if len(fail_flows) == 0:                                # 成功重路由, 认为done
            done = True
        if fail_flows == last_fail_flows:                       # 做出动作没任何效果, 认为done, 且由于没做出改变, 所以直接continue
            done = True
    print(f"greed_maxsum_reward: \t{total_reward:.2f}", end=", \t")
    return total_reward

################### 正式流程代码 ###################
try:
    print_current_time("start")

    # 初始化覆盖txt
    with open(save_dir_output + "losses.txt", "w") as f:
        f.write(f"")
    with open(save_dir_output + "rewards.txt", "w") as f:
        f.write(f"")
    with open(save_dir_output + "rewards_vs.txt", "w") as f:
        f.write(f"")
    with open(save_dir_output + "time_vs.txt", "w") as f:
        f.write(f"")

    # 读入拓扑
    graph = myClass.m_graph()
    init_env_actions = graph.initial_generate_ba(100, 2)
    print(f"n = {graph.n}, m = {graph.m}, f = {graph.f}")
    print(f"init_env_actions: \n{init_env_actions}")

    # 初始化模型
    model = gnnLyx(hparams).to(device)
    model_state = torch.load("model_epoch_300000.pth", map_location=device)
    model.load_state_dict(model_state)
    # torch.autograd.set_detect_anomaly(True)             # 调试时开启

    print(f"model_device: {next(model.parameters()).device}") # 输出：cpu 或 cuda:0

    # 初始化变量
    n, m, flow_cnt = graph.n, graph.m, graph.f
    k = myClass.m_graph.K_SP_CNT

    # 初始化参数
    episodes = 1000000     # 跑多少轮
    FAIL_LINK_CNT_MIN, FAIL_LINK_CNT_MAX = 1, 3
    FAIL_FLOW_CNT_MIN, FAIL_FLOW_CNT_MAX = 2, 5

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
            fail_links = []
            for _ in range(fail_links_cnt):
                while True:
                    link_id = random.randint(0, m - 1)
                    if link_id not in fail_links:
                        fail_links.append(link_id)
                        break
            fail_flows = graph.get_fail_flows(env_actions, fail_links)
            if len(fail_flows) >= FAIL_FLOW_CNT_MIN and len(fail_flows) <= FAIL_FLOW_CNT_MAX:
                break
        
        print(f"\nnow_episode: {now_episode}")
            
        start = time.perf_counter()# 计时--------------------------------------------------------------
        reward_model, actions_cnt = get_reward_model(graph, env_actions, fail_links)
        time_model = (time.perf_counter() - start) * 1000
        print(f"耗时: \t{(time.perf_counter() - start) * 1000:.3f} 毫秒")

        start = time.perf_counter()# 计时--------------------------------------------------------------
        reward_random = get_reward_random(graph, env_actions, fail_links)
        time_random = (time.perf_counter() - start) * 1000
        print(f"耗时: \t{(time.perf_counter() - start) * 1000:.3f} 毫秒")

        start = time.perf_counter()# 计时--------------------------------------------------------------
        reward_greed_sum = get_reward_greed_sum(graph, env_actions, fail_links)
        time_sum = (time.perf_counter() - start) * 1000
        print(f"耗时: \t{(time.perf_counter() - start) * 1000:.3f} 毫秒")

        start = time.perf_counter()# 计时--------------------------------------------------------------
        reward_greed_max = get_reward_greed_min(graph, env_actions, fail_links)
        time_max = (time.perf_counter() - start) * 1000
        print(f"耗时: \t{(time.perf_counter() - start) * 1000:.3f} 毫秒")

        start = time.perf_counter()# 计时--------------------------------------------------------------
        reward_greed_max_sum = get_reward_greed_max_sum(graph, env_actions, fail_links)
        time_max_sum = (time.perf_counter() - start) * 1000
        print(f"耗时: \t{(time.perf_counter() - start) * 1000:.3f} 毫秒")

        # 此处为一轮训练完毕
        with open(save_dir_output + "rewards_vs.txt", "a") as f:
                f.write(f"{now_episode},{reward_model},{reward_random},{reward_greed_sum},{reward_greed_max},{reward_greed_max_sum}\n")
        with open(save_dir_output + "time_vs.txt", "a") as f:
                f.write(f"{actions_cnt},{time_model},{time_random},{time_sum},{time_max},{time_max_sum}\n")

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    print_current_time("end")
    print("Done")