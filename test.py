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
memory = deque(maxlen=2000)
batch_size = 32
target_model_update_freq = 1000      # 目标网络更新频率
target_model_save_freq = 5000        # 目标网络保存频率
reward_gamma = 0.95     # reward 计算参数
eval_flag = False       # True: 推理模式,   False: 训练模式

# 贪心参数
epsilon = 1.0
epsilon_min = 0.001
epsilon_decay = 0.995
# 当前需要ln(0.001/1.0)/ln(0.995) = 1378轮

# 记录list
rewards = []
losses = []

################### 函数代码 ###################
def make_exprience(graph, fail_links, env_actions, reward, done):
    eval_link_attr, eval_path_attr, eval_mask = graph.get_features_one(fail_links, env_actions)
    return eval_link_attr, eval_path_attr, eval_mask, reward, done

def print_current_time():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"start time : {current_time}:\n")

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

################### 正式流程代码 ###################
try:
    print_current_time()

    # 初始化覆盖txt
    with open(save_dir_output + "losses.txt", "w") as f:
        f.write(f"")
    with open(save_dir_output + "rewards.txt", "w") as f:
        f.write(f"")

    # 读入拓扑
    graph = myClass.m_graph()
    init_env_actions = graph.initial_generate_ba(100, 2)
    print(f"n = {graph.n}, m = {graph.m}, f = {graph.f}")
    print(f"init_env_actions: \n{reprlib.repr(init_env_actions)}")

    # 初始化模型
    model = gnnLyx(hparams).to(device)
    target_model = gnnLyx(hparams).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)    # 这里设置学习率
    if (eval_flag):
        model_state = torch.load("model_epoch_300000.pth", map_location=device)
        model.load_state_dict(model_state)
    target_model.load_state_dict(model.state_dict())    # 初始参数相同
    # torch.autograd.set_detect_anomaly(True)             # 调试时开启

    print(f"model_device: {next(model.parameters()).device}") # 输出：cpu 或 cuda:0

    # 初始化变量
    n, m, flow_cnt = graph.n, graph.m, graph.f
    k = myClass.m_graph.K_SP_CNT

    # 初始化参数
    episodes = 1000000     # 跑多少轮
    total_step = 0
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
        
        # 失效环境初始化成功
        total_bw = sum([graph.flows[flow_id].bw for flow_id in range(flow_cnt)])
        origin_throughput = sum(graph.get_path_throughput(env_actions, fail_flows))
        stepIdx = 0
        reward = 0
        total_reward = 0
        done = False
        while not done:
            total_step += 1
            stepIdx += 1

            print(f"\nTotal_step: {total_step}, Memory_len = {len(memory)} --------------------------------")

            last_env_actions = env_actions
            last_fail_flows = fail_flows

            # ################### 推理模式 ###################
            start = time.perf_counter()# 计时--------------------------------------------------------------
            if np.random.rand() < epsilon and eval_flag == False:
                print("Random Process")
                _, _, _, new_actions_list = graph.get_features(env_actions, fail_flows, device)
                env_actions = new_actions_list[random.randint(0, len(new_actions_list) - 1)]
            else:
                model.eval()
                with torch.no_grad():
                    link_attr, path_attr, mask, new_actions_list = graph.get_features(env_actions, fail_flows, device)
                    q_values = model(link_attr, path_attr, mask)
                    max_q_value, max_q_index = torch.max(q_values, dim=0)
                    best_actions_index = max_q_index.item()
                    env_actions = new_actions_list[best_actions_index]
            
            fail_flows = graph.get_fail_flows(env_actions, fail_links)
            new_throughput = sum(graph.get_path_throughput(env_actions, fail_flows))
            reward = new_throughput / total_bw
            performance = (new_throughput - origin_throughput) / (total_bw - origin_throughput + 1e-6)   # 防止除0

            print(f"决策部分耗时: {(time.perf_counter() - start) * 1000:.3f} 毫秒")
            print(f"old\tactions: \t{reprlib.repr(last_env_actions)}")
            print(f"\tfail_flows: \t{last_fail_flows}")
            print(f"new\tactions: \t{reprlib.repr(env_actions)}")
            print(f"\tfail_flows: \t{fail_flows}")
            print(f"reward\t\t = {reward:.4f}\t = {new_throughput:.2f} / {total_bw:.2f}")
            print(f"performance\t = {performance:.4f}\t = ({new_throughput:.2f} - {origin_throughput:.2f}) / ({total_bw:.2f} - {origin_throughput:.2f})")

            done = True    # 改成每轮只进行一步决策

            # 记录经验池 (s, a, s', r, done)
            # 由于发现提取特征值比较慢，现在改成了直接传特征值
            start = time.perf_counter()# 计时--------------------------------------------------------------
            experience = make_exprience(graph, fail_links, env_actions, reward, done)
            memory.append(copy.deepcopy(experience))
            print(f"经验记录耗时: {(time.perf_counter() - start) * 1000:.3f} 毫秒")

            # ################### 训练模式 ###################
            start = time.perf_counter()# 计时--------------------------------------------------------------
            if len(memory) > batch_size and eval_flag == False:
                model.train()
                target_model.eval()

                batch = random.sample(memory, batch_size)
                eval_link_attr, eval_path_attr, eval_mask, exp_rewards, exp_done = zip(*batch)
                

                # 先获取eval的q值
                eval_q_values = model(torch.tensor(eval_link_attr, device=device), torch.tensor(eval_path_attr, device=device), torch.tensor(eval_mask, device=device))

                # 再获取target的q值
                target_q_values = torch.tensor(exp_rewards, device=device)

                print(f"eval_q: {reprlib.repr(eval_q_values.tolist())}\ntarg_q: {reprlib.repr(target_q_values.tolist())}")

                loss = nn.functional.mse_loss(eval_q_values, target_q_values.detach())
                print(f"loss = {loss.item()}")

                optimizer.zero_grad()# 清除梯度
                loss.backward()# 反向传播
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)# 梯度裁剪 (可选)
                optimizer.step()# 参数更新

                # 记录损失
                losses.append(loss.item())
                with open(save_dir_output + "losses.txt", "a") as f:
                    f.write(f"{total_step},{loss}\n")
            
            print(f"训练部分耗时: {(time.perf_counter() - start) * 1000:.3f} 毫秒")
            
            # 同步参数到另一个网络
            if total_step % target_model_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
            # 保存网络
            if total_step % target_model_save_freq == 0:
                # 保存模型
                save_path = os.path.join(save_dir_models, f'model_epoch_{total_step}.pth') #保存模型，位置在开头定义,要注意执行路径
                torch.save(target_model.state_dict(), save_path)
                print(f'Model saved at {save_path}')
            
            # 减少epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

        # 此处为一轮训练完毕
        print(f"episode: {now_episode}, reward = {reward:.4f}, performance = {performance:.4f}, epsilon = {epsilon:.6f}")
        with open(save_dir_output + "rewards.txt", "a") as f:
                f.write(f"{total_step},{reward}\n")
        with open(save_dir_output + "performance.txt", "a") as f:
                f.write(f"{total_step},{performance}\n")

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    print_current_time()
    print("Done")