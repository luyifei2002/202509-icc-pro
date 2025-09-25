#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
target_model_update_freq = 100      # 目标网络更新频率
target_model_save_freq = 500        # 目标网络保存频率
reward_gamma = 0.95     # reward 计算参数
eval_flag = False       # True: 推理模式,   False: 训练模式

# 贪心参数
epsilon = 0.9
epsilon_init = 0.9
epsilon_min = 0.0005
epsilon_decay = 0.99

# 记录list
rewards = []
losses = []

################### 正式流程代码 ###################
try:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"start time : {current_time}:\n")
    # 初始化覆盖txt
    with open(save_dir_output + "losses.txt", "w") as f:
        f.write(f"")
    with open(save_dir_output + "rewards.txt", "w") as f:
        f.write(f"")

    # 读入拓扑
    graph = myClass.m_graph()
    init_env_actions = graph.initial_generate_ba(10, 2)
    print(f"n = {graph.n}, m = {graph.m}, f = {graph.f}")
    print(f"init_env_actions: \n{init_env_actions}")

    # 初始化模型
    model = gnnLyx(hparams).to(device)
    target_model = gnnLyx(hparams).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)    # 这里设置学习率
    if (eval_flag):
        model_state = torch.load("model_epoch_20000.pth", map_location=device)
        model.load_state_dict(model_state)
    target_model.load_state_dict(model.state_dict())    # 初始参数相同
    # torch.autograd.set_detect_anomaly(True)             # 调试时开启

    print(f"model_device: {next(model.parameters()).device}") # 输出：cpu 或 cuda:0

    # 初始化变量
    n, m, flow_cnt = graph.n, graph.m, graph.f
    k = myClass.m_graph.K_SP_CNT

    # 初始化参数
    episodes = 2000     # 跑多少轮
    total_step = 0
    FAIL_LINK_CNT_MIN, FAIL_LINK_CNT_MAX = 2, 5
    FAIL_FLOW_CNT_MIN, FAIL_FLOW_CNT_MAX = 3, 6

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
        
        # 如果是训练模式，初始化贪心epsilon
        if eval_flag == False:
            epsilon = epsilon_init
        else:
            epsilon = 0
        stepIdx = 0
        reward = 0
        total_reward = 0
        done = False
        while not done:
            total_step += 1
            stepIdx += 1

            print(f"\nStepIdx: {stepIdx}, Total_step: {total_step}, Total_reward = {total_reward}")

            last_env_actions = env_actions
            last_fail_flows = fail_flows

            # ################### 推理模式 ###################
            start = time.perf_counter()# 计时--------------------------------------------------------------
            model.eval()
            if np.random.rand() < epsilon and eval_flag == False:
                for flow_id in fail_flows:
                    path_cnt = min(k, len(graph.flows[flow_id].paths))
                    env_actions[flow_id] = random.randint(0, path_cnt - 1)
                print("Random Process")
            else:
                with torch.no_grad():
                    features, new_actions_list = graph.get_features(env_actions, fail_flows, device)
                    # print(f"features: \nlink_attr: \n{features['link_attr'].tolist()}, \npath_attr: \n{features['path_attr'].tolist()}, \nmask: \n{features['mask'].tolist()}")
                    q_values = model(features)
                    max_q_value, max_q_index = torch.max(q_values, dim=0)
                    best_actions_index = max_q_index.item()
                    env_actions = new_actions_list[best_actions_index]
            
            fail_flows = graph.get_fail_flows(env_actions, fail_links)
            reward = 1.0 * (len(last_fail_flows) - len(fail_flows))
            if len(fail_flows) > FAIL_FLOW_CNT_MAX:                 # 操作后失效的比原来多多了, 认为done
                done = True
                reward = -1.0 * len(fail_flows)
            if len(fail_flows) == 0:                                # 成功重路由, 认为done
                done = True
            if total_reward < -1.0 * flow_cnt:                      # 超出阈值, 认为done
                done = True
            if fail_flows == last_fail_flows:                       # 做出动作没任何效果, 认为done, 且由于没做出改变, 所以直接continue
                done = True
                continue
            total_reward += reward
            
            print(f"决策部分耗时: {(time.perf_counter() - start) * 1000:.3f} 毫秒")
            start = time.perf_counter()# 计时--------------------------------------------------------------
            print(f"old\tactions: \t\t{last_env_actions}")
            print(f"\tfail_flows: \t{last_fail_flows}")
            print(f"new\tactions: \t\t{env_actions}")
            print(f"\tfail_flows: \t{fail_flows}")

            # 记录经验池 (s, a, s', r, done)
            # [fail_links, last_env_actions] = s    # 似乎训练中不必要
            # [fail_links, env_actions] = s + a = s'
            experience = (fail_links, env_actions, reward, done)
            memory.append(copy.deepcopy(experience))

            # ################### 训练模式 ###################
            if len(memory) > batch_size and eval_flag == False:
                model.train()
                target_model.eval()

                batch = random.sample(memory, batch_size)
                exp_fail_links, exp_env_actions, exp_rewards, exp_done = zip(*batch)

                # 先获取eval的q值
                eval_features = graph.get_features_tuple(exp_fail_links, exp_env_actions, device)
                eval_q_values = model(eval_features)

                # 再获取target的q值
                target_q_values_list = []
                with torch.no_grad():
                    for i in range(len(batch)):
                        target_env_actions = exp_env_actions[i]
                        target_fail_links = exp_fail_links[i]
                        target_reward = exp_rewards[i]
                        target_done = exp_done[i]

                        if target_done:
                            target_q_values_list.append(target_reward)
                            continue

                        target_fail_flows = graph.get_fail_flows(target_env_actions, target_fail_links)
                        target_features, target_new_actions_list = graph.get_features(target_env_actions, target_fail_flows, device)
                        target_q_values = target_model(target_features)
                        target_max_q_value, target_max_q_index = torch.max(target_q_values, dim=0)
                        target_best_value = target_max_q_value.item()
                        target_q_value = target_reward + reward_gamma * target_best_value

                        target_q_values_list.append(target_q_value)
                target_q_values = torch.tensor(target_q_values_list, device=device)

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
            start = time.perf_counter()# 计时--------------------------------------------------------------
            
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
        print(f"episode: {now_episode}, total_reward = {total_reward}")
        with open(save_dir_output + "rewards.txt", "a") as f:
                f.write(f"{total_step},{total_reward}\n")

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    print("Done")