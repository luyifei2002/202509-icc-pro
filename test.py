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
epsilon = 1.0
epsilon_min = 0.001
epsilon_decay = 0.995
# 当前需要ln(0.001/1.0)/ln(0.995) = 1378轮

# 记录list
rewards = []
losses = []

################### 函数代码 ###################
def make_exprience(graph, fail_links, env_actions, reward, done):
    eval_link_attr, eval_path_attr, eval_mask = graph.get_features_tuple_one(fail_links, env_actions)
    target_link_attr, target_path_attr, target_mask = [], [], []
    if not done:
        target_fail_flows = graph.get_fail_flows(env_actions, fail_links)
        target_link_attr, target_path_attr, target_mask = graph.get_features_target_exprience(env_actions, target_fail_flows)
    return eval_link_attr, eval_path_attr, eval_mask, target_link_attr, target_path_attr, target_mask, reward, done


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
    init_env_actions = graph.initial_generate_ba(100, 2)
    print(f"n = {graph.n}, m = {graph.m}, f = {graph.f}")
    print(f"init_env_actions: \n{init_env_actions}")

    # 初始化模型
    model = gnnLyx(hparams).to(device)
    target_model = gnnLyx(hparams).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)    # 这里设置学习率
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
    FAIL_FLOW_CNT_MIN, FAIL_FLOW_CNT_MAX = 1, 5

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
            if np.random.rand() < epsilon and eval_flag == False:
                for flow_id in fail_flows:
                    path_cnt = min(k, len(graph.flows[flow_id].paths))
                    env_actions[flow_id] = random.randint(0, path_cnt - 1)
                print("Random Process")
            else:
                model.eval()
                with torch.no_grad():
                    link_attr, path_attr, mask, new_actions_list = graph.get_features(env_actions, fail_flows, device)
                    q_values = model(link_attr, path_attr, mask)
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
            print(f"old\tactions: \t{last_env_actions}")
            print(f"\tfail_flows: \t{last_fail_flows}")
            print(f"new\tactions: \t{env_actions}")
            print(f"\tfail_flows: \t{fail_flows}")

            # 记录经验池 (s, a, s', r, done)
            # 由于发现提取特征值比较慢，现在改成了直接传特征值
            experience = make_exprience(graph, fail_links, env_actions, reward, done)
            memory.append(copy.deepcopy(experience))

            # ################### 训练模式 ###################
            start = time.perf_counter()# 计时--------------------------------------------------------------
            if len(memory) > batch_size and eval_flag == False:
                model.train()
                target_model.eval()

                batch = random.sample(memory, batch_size)
                eval_link_attr, eval_path_attr, eval_mask, exp_link_attr, exp_path_attr, exp_mask, exp_rewards, exp_done = zip(*batch)
                
                start1 = time.perf_counter()# 计时------------

                # 先获取eval的q值
                eval_q_values = model(torch.tensor(eval_link_attr, device=device), torch.tensor(eval_path_attr, device=device), torch.tensor(eval_mask, device=device))

                print(f"eval_q_values\t计算耗时: {(time.perf_counter() - start1) * 1000:.3f} 毫秒")
                start1 = time.perf_counter()# 计时------------

                # 再获取target的q值
                target_q_values_list = []
                with torch.no_grad():
                    target_len_batch = [len(link_attr) for link_attr in exp_link_attr]
                    target_total_len = sum(target_len_batch)
                    target_link_attr_arr = np.zeros((target_total_len, m, 4), dtype=np.float32)
                    target_path_attr_arr = np.zeros((target_total_len, flow_cnt, 1), dtype=np.float32)
                    target_mask_arr = np.full((target_total_len, flow_cnt, m), False, dtype=np.bool_)

                    start2 = time.perf_counter()# 计时------------
                    target_l = 0
                    for i in range(len(batch)):
                        target_len = target_len_batch[i]
                        if target_len == 0:
                            continue
                        target_link_attr_arr_temp = np.array(exp_link_attr[i], dtype=np.float32).reshape(target_len, m, 4)
                        target_path_attr_arr_temp = np.array(exp_path_attr[i], dtype=np.float32).reshape(target_len, flow_cnt, 1)
                        target_mask_arr_temp = np.array(exp_mask[i], dtype=np.bool_).reshape(target_len, flow_cnt, m)
                        target_link_attr_arr[target_l:target_l + target_len] = target_link_attr_arr_temp
                        target_path_attr_arr[target_l:target_l + target_len] = target_path_attr_arr_temp
                        target_mask_arr[target_l:target_l + target_len] = target_mask_arr_temp
                        target_l += target_len

                    print(f"\t\t\t合并\t计算耗时: {(time.perf_counter() - start2) * 1000:.3f} 毫秒")
                    start2 = time.perf_counter()# 计时------------

                    target_q_values_batch = target_model(torch.as_tensor(target_link_attr_arr, device=device), torch.as_tensor(target_path_attr_arr, device=device), torch.as_tensor(target_mask_arr, device=device))
                    target_q_values_oringin_list = target_q_values_batch.tolist()

                    print(f"\t\t\t模型\t计算耗时: {(time.perf_counter() - start2) * 1000:.3f} 毫秒")
                    start2 = time.perf_counter()# 计时------------

                    l = 0
                    for i in range(len(batch)):
                        target_reward = exp_rewards[i]
                        target_done = exp_done[i]
                        if target_done:
                            target_q_values_list.append(target_reward)
                            continue
                        target_max_q_value = max([target_q_values_oringin_list[l + j] for j in range(target_len_batch[i])])
                        target_q_value = target_reward + reward_gamma * target_max_q_value
                        target_q_values_list.append(target_q_value)
                        l += target_len_batch[i]
                    print(f"\t\t\tQ值\t计算耗时: {(time.perf_counter() - start2) * 1000:.3f} 毫秒")

                target_q_values = torch.tensor(target_q_values_list, device=device)

                print(f"target_q_values\t计算耗时: {(time.perf_counter() - start1) * 1000:.3f} 毫秒")
                start1 = time.perf_counter()# 计时------------

                loss = nn.functional.mse_loss(eval_q_values, target_q_values.detach())

                optimizer.zero_grad()# 清除梯度
                loss.backward()# 反向传播
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)# 梯度裁剪 (可选)
                optimizer.step()# 参数更新

                print(f"loss\t\t计算耗时: {(time.perf_counter() - start1) * 1000:.3f} 毫秒")

                print(f"loss = {loss.item()}")
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
        print(f"episode: {now_episode}, total_reward = {total_reward}")
        with open(save_dir_output + "rewards.txt", "a") as f:
                f.write(f"{total_step},{total_reward}\n")

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    print("Done")