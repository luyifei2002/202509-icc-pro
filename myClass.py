from collections import deque
import numpy as np
import math
import heapq
import copy
import torch
import networkx as nx
import random

class m_node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class m_edge:
    def __init__(self, u, v, next_head, next_tail, bw, delay):
        self.u = u
        self.v = v
        self.next_head = next_head
        self.next_tail = next_tail
        self.bw = bw
        self.delay = delay

class m_flow:
    def __init__(self, s, t, bw):
        self.s = s
        self.t = t
        self.bw = bw
        self.paths = []

class m_ksp_node1:
    def __init__(self, v, d):
        self.v = v
        self.d = d

    # 定义小于运算符，用于堆排序
    def __lt__(self, other):
        return self.d < other.d

class m_ksp_node2:
    def __init__(self, v, d, H, last_path:list):
        self.v = v
        self.d = d
        self.H = H
        self.path = copy.deepcopy(last_path)
        self.path.append(v)

    # 定义小于运算符，用于堆排序
    def __lt__(self, other):
        return self.d + self.H[self.v] < other.d + self.H[other.v]

class m_graph:
    K_SP_CNT = 4 # 最短路数量参数
    LINK_BW_MIN = 6
    LINK_BW_MAX = 12
    LINK_DELAY_MIN = 100
    LINK_DELAY_MAX = 300
    FLOW_BW_MIN = 2
    FLOW_BW_MAX = 4
    FLOW_DIJKSTRA_REDUNDANT = 0.3

    def __init__(self):
        self.n = 0
        self.m = 0
        self.f = 0
        self.nodes = []
        self.edges = []
        self.edge_head = []
        self.edge_tail = []
        self.flows = []

        # 介数相关变量
        self.jieshu_vis = []
        self.jieshu_dis = []
        self.jieshu_cnt = []
        self.jieshu_ret = []

    def add_edge(self, u, v, bw, delay):
        self.edges.append(m_edge(u, v, self.edge_head[u], self.edge_tail[v], bw, delay))
        self.edge_head[u] = len(self.edges) - 1
        self.edge_tail[v] = len(self.edges) - 1

    def reset(self):
        self.n = 0
        self.m = 0
        self.f = 0
        self.nodes = []
        self.edges = []
        self.edge_head = []
        self.edge_tail = []
        self.flows = []
        self.H = []
        self.jieshu_vis = []
        self.jieshu_dis = []
        self.jieshu_cnt = []
        self.jieshu_ret = []

    def readin(self, file_name):
        path_dir = "./"
        file_dir = path_dir + file_name
        self.reset()
        try:
            with open(file_dir, 'r') as file:
                line = file.readline().strip()
                n, m, f = map(int, line.split())
                self.n = n
                self.m = m
                self.f = f
                self.edge_head = [-1 for _ in range(n)]
                self.edge_tail = [-1 for _ in range(n)]
                self.H = [65535 for _ in range(n)]
                self.jieshu_vis = [[0 for _ in range(self.n)] for _ in range(self.n)]
                self.jieshu_dis = [[np.inf for _ in range(self.n)] for _ in range(self.n)]
                self.jieshu_cnt = [[0 for _ in range(self.n)] for _ in range(self.n)]

                for _ in range(n):
                    line = file.readline().strip().split()
                    x, y, z = map(float, line)
                    self.nodes.append(m_node(x, y, z))

                for _ in range(m):
                    line = file.readline().strip().split()
                    u = int(line[0])
                    v = int(line[1])
                    bw = float(line[2])
                    dis = self.cal_node_dis(u, v)
                    self.add_edge(u, v, bw, dis / 300)

                for _ in range(f):
                    line = file.readline().strip().split()
                    s = int(line[0])
                    t = int(line[1])
                    bw = float(line[2])
                    self.flows.append(m_flow(s, t, bw))

        except Exception as e:
            print(f"读取文件时发生错误: {e}")

    def cal_node_dis(self, u, v):
        n1 = self.nodes[u]
        n2 = self.nodes[v]
        return math.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2 + (n1.z - n2.z) ** 2)

    def cal_k_sp(self, flow_id):
        s = self.flows[flow_id].s
        t = self.flows[flow_id].t
        bw = self.flows[flow_id].bw
        # 第一遍dijkstra
        heap1 = [m_ksp_node1(t, 0)]
        vis = [False for _ in range(self.n)]
        while heap1:
            node_now = heapq.heappop(heap1)
            u, d = node_now.v, node_now.d
            if vis[u]:
                continue
            vis[u] = True
            self.H[u] = d
            # 遍历每一个邻居
            e_id = self.edge_head[u]
            while e_id != -1:
                edge = self.edges[e_id]
                heapq.heappush(heap1, m_ksp_node1(edge.v, d + edge.delay))
                e_id = edge.next_head
            e_id = self.edge_tail[u]
            while e_id != -1:
                edge = self.edges[e_id]
                heapq.heappush(heap1, m_ksp_node1(edge.u, d + edge.delay))
                e_id = edge.next_tail
        # 第二遍dijkstra
        heap2 = [m_ksp_node2(s, 0, self.H, [])]
        cnt = [0 for _ in range(self.n)]
        while heap2:
            node_now = heapq.heappop(heap2)
            u, d, path = node_now.v, node_now.d, node_now.path
            cnt[u] += 1
            if u == t and cnt[u] <= m_graph.K_SP_CNT:
                self.flows[flow_id].paths.append(copy.deepcopy(path))
            if cnt[u] > m_graph.K_SP_CNT:
                continue
            e_id = self.edge_head[u]
            while e_id != -1:
                edge = self.edges[e_id]

                # 如果下一步在已经走过的path里，就跳过
                vis_flag = False
                for v_temp in path:
                    if v_temp == edge.v:
                        vis_flag = True
                        break
                if vis_flag == False:
                    heapq.heappush(heap2, m_ksp_node2(edge.v, d + edge.delay, self.H, path))
                e_id = edge.next_head
            e_id = self.edge_tail[u]
            while e_id != -1:
                edge = self.edges[e_id]

                # 如果下一步在已经走过的path里，就跳过
                vis_flag = False
                for v_temp in path:
                    if v_temp == edge.u:
                        vis_flag = True
                        break
                if vis_flag == False:
                    heapq.heappush(heap2, m_ksp_node2(edge.u, d + edge.delay, self.H, path))
                e_id = edge.next_tail

    def initial(self, file_name):
        self.reset()
        self.readin(file_name)
        for i in range(self.f):
            self.cal_k_sp(i)
        self.cal_jieshu()
    
    def initial_generate_ba(self, n, m):    # 生成一个ba无标度网络拓扑, 参数为节点数n, 每次加点连边数m
        G = nx.barabasi_albert_graph(n, m=m)

        print(f"n = {G.number_of_nodes()}, m = {G.number_of_edges()}")
        print("平均最短路径长度为：",nx.average_shortest_path_length(G))

        # 初始化参数
        self.n = G.number_of_nodes()
        self.m = G.number_of_edges()
        self.f = 0
        self.edge_head = [-1 for _ in range(n)]
        self.edge_tail = [-1 for _ in range(n)]
        self.H = [65535 for _ in range(n)]
        self.jieshu_vis = [[0 for _ in range(self.n)] for _ in range(self.n)]
        self.jieshu_dis = [[np.inf for _ in range(self.n)] for _ in range(self.n)]
        self.jieshu_cnt = [[0 for _ in range(self.n)] for _ in range(self.n)]

        # 添加边, 备注: 生成图这一块不考虑节点位置属性了, 边delay随机生成
        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        for u, v in G.edges():
            src = node_mapping.get(u)
            dst = node_mapping.get(v)
            bw = random.randint(self.LINK_BW_MIN, self.LINK_BW_MAX)
            delay = random.randint(self.LINK_DELAY_MIN, self.LINK_DELAY_MAX)

            self.add_edge(src, dst, bw, delay)
        
        # 添加业务流, 备注: 考虑一直加流到加不了
        cant_add_flag = 0   # 加流连续失败这么多次就认为加不了
        link_bw_used = [0 for _ in range(self.m)]
        env_actions = []    # 记录选择的路径
        while cant_add_flag <= self.n:
            src = random.randint(0, self.n - 1)
            dst = random.randint(0, self.n - 1)
            if src == dst:
                continue
            bw = random.randint(self.FLOW_BW_MIN, self.FLOW_BW_MAX)

            # 带约束寻最短路
            vis = [False for _ in range(self.n)]    # 访问标志
            dis = [1e30 for _ in range(self.n)]     # 距离记录
            last = [-1 for _ in range(self.n)]      # 路径记录
            q = deque()
            q.append(src)
            dis[src] = 0
            while q:
                u = q.pop()
                vis[u] = False

                e_id = self.edge_head[u]
                while e_id != -1:
                    edge = self.edges[e_id]
                    v = edge.v
                    if dis[v] <= dis[u] + edge.delay or edge.bw * (1 - self.FLOW_DIJKSTRA_REDUNDANT) < link_bw_used[e_id] + bw:
                        e_id = edge.next_head
                        continue
                    dis[v] = dis[u] + edge.delay
                    last[v] = u
                    if not vis[v]:
                        vis[v] = True
                        q.append(v)
                    e_id = edge.next_head

                e_id = self.edge_tail[u]
                while e_id != -1:
                    edge = self.edges[e_id]
                    v = edge.u
                    if dis[v] <= dis[u] + edge.delay or edge.bw * (1 - self.FLOW_DIJKSTRA_REDUNDANT) < link_bw_used[e_id] + bw:
                        e_id = edge.next_tail
                        continue
                    dis[v] = dis[u] + edge.delay
                    last[v] = u
                    if not vis[v]:
                        vis[v] = True
                        q.append(v)
                    e_id = edge.next_tail
            if dis[dst] == 1e30:    # 找不到可行最短路
                cant_add_flag += 1
                continue
            now_u = dst
            now_path = [now_u]
            while True:
                now_v = last[now_u]
                if now_v == -1:
                    break
                e_id = self.get_edgeId_by_node(now_u, now_v)
                link_bw_used[e_id] += bw
                now_path.append(now_v)
                now_u = now_v
            self.flows.append(m_flow(src, dst, bw))
            self.cal_k_sp(self.f)
            now_path.reverse()
            choose_flag = False
            for path_id in range(len(self.flows[self.f].paths)):
                path = self.flows[self.f].paths[path_id]
                if len(path) != len(now_path):
                    continue
                same_flag = True
                for i in range(len(path)):
                    if path[i] != now_path[i]:
                        same_flag = False
                        break
                if same_flag:
                    choose_flag = True
                    env_actions.append(path_id)
                    break
            if not choose_flag:     # 该流还是不可行
                self.flows.pop()
                cant_add_flag += 1
                continue
            cant_add_flag = 0
            self.f += 1
        self.cal_jieshu()
        return env_actions

    def cal_jieshu_bfs(self, s):
        vis = self.jieshu_vis[s]
        dis = self.jieshu_dis[s]
        cnt = self.jieshu_cnt[s]
        vis[s] = 1
        dis[s] = 0
        cnt[s] = 1
        q = deque()
        q.append(s)
        while(q):
            u = q.popleft()

            e_id = self.edge_head[u]
            while e_id != -1:
                edge = self.edges[e_id]
                v = edge.v
                if vis[v] == 1:
                    if dis[v] == dis[u] + 1:
                        cnt[v] += cnt[u]
                    e_id = edge.next_head
                    continue
                vis[v] = 1
                dis[v] = dis[u] + 1
                cnt[v] = cnt[u]
                q.append(v)
                e_id = edge.next_head

            e_id = self.edge_tail[u]
            while e_id != -1:
                edge = self.edges[e_id]
                v = edge.u
                if vis[v] == 1:
                    if dis[v] == dis[u] + 1:
                        cnt[v] += cnt[u]
                    e_id = edge.next_tail
                    continue
                vis[v] = 1
                dis[v] = dis[u] + 1
                cnt[v] = cnt[u]
                q.append(v)
                e_id = edge.next_tail

    def cal_jieshu(self):
        self.jieshu_ret = []

        # 最短路计数
        for i in range(self.n):
            self.cal_jieshu_bfs(i)

        # 计算介数
        for edge in self.edges:
            jieshu = 0.000
            u, v = edge.u, edge.v
            for j in range(self.n):
                if j == u or j == v:
                    continue
                for k in range(self.n):
                    if k == u or k == v:
                        continue
                    if self.jieshu_dis[j][u] + self.jieshu_dis[v][k] + 1 == self.jieshu_dis[j][k]:
                        jieshu += 1.0 * self.jieshu_cnt[j][u] * self.jieshu_cnt[v][k] / self.jieshu_cnt[j][k]
            self.jieshu_ret.append(jieshu)

    def print_all_flow_all_path(self):
        for i in range(self.f):
            print(f'flow {i} : ')
            flow = self.flows[i]
            for j in range(min(m_graph.K_SP_CNT, len(flow.paths))):
                print(f'{flow.paths[j]}')

    def get_edgeId_by_node(self, u, v):
        e_id = self.edge_head[u]
        while e_id != -1:
            edge = self.edges[e_id]
            if edge.v == v:
                return e_id
            e_id = edge.next_head
        e_id = self.edge_tail[u]
        while e_id != -1:
            edge = self.edges[e_id]
            if edge.u == v:
                return e_id
            e_id = edge.next_tail

    def get_link_capacity(self):
        return [edge.bw for edge in self.edges]

    def get_link_capacity_available(self, env_actions):
        link_capacity = [edge.bw for edge in self.edges]
        for i in range(self.f):
            flow = self.flows[i]
            path = flow.paths[env_actions[i]]
            for j in range(len(path) - 1):
                link_id = self.get_edgeId_by_node(path[j], path[j + 1])
                link_capacity[link_id] -= flow.bw
        return link_capacity

    def get_link_attr_max_fail_p_list(self, env_actions, fail_flows):
        # 两条公理：
        # 1. fail_path上至少有一条fail_link
        # 2. active_path上必定全部为active_link
        link_fail_cnt = [0 for _ in range(self.m)]                          # 记录每个可能的fail_link上经过的fail_path的数量, 若值为-1表示必定为active_link
        link_attr_max_fail_p = [0 for _ in range(self.m)]                   # 按link取最大值作为该link的失效概率的特征

        # 1、确定所有path
        is_fail_path = [0 for _ in range(self.f)]
        for id in range(len(fail_flows)):
            is_fail_path[id] = 1

        # 2、确定所有active_link
        for i in range(self.f):
            if is_fail_path[i]:
                continue
            active_path = self.flows[i].paths[env_actions[i]]
            for j in range(len(active_path) - 1):
                link_id = self.get_edgeId_by_node(active_path[j], active_path[j + 1])
                link_fail_cnt[link_id] = -1

        # 3、计算link_fail_cnt
        for i in range(self.f):
            if not is_fail_path[i]:
                continue
            fail_path = self.flows[i].paths[env_actions[i]]
            for j in range(len(fail_path) - 1):
                link_id = self.get_edgeId_by_node(fail_path[j], fail_path[j + 1])
                if link_fail_cnt[link_id] != -1:
                    link_fail_cnt[link_id] += 1

        # 4、计算每个fail_path的softmax(link_fail_cnt)作为每个fail_path中的每条link的失效概率
        for i in range(self.f):
            if not is_fail_path[i]:
                continue
            fail_path = self.flows[i].paths[env_actions[i]]
            path_fail_cnt = [0 for _ in range(self.m)]
            for j in range(len(fail_path) - 1):
                link_id = self.get_edgeId_by_node(fail_path[j], fail_path[j + 1])
                if link_fail_cnt[link_id] != -1:
                    path_fail_cnt[link_id] = link_fail_cnt[link_id]
                else: path_fail_cnt[link_id] = -1e30  # 当作-inf
            path_fail_cnt_array = np.array(path_fail_cnt)
            e_x = np.exp(path_fail_cnt_array - np.max(path_fail_cnt_array))
            softmax_x = e_x / np.sum(e_x)
            path_fail_softmax = softmax_x.tolist()

            # 5、最后每条边取fail_path计算出的fail_P的最大值, 以此量化每个link的失效可能性
            link_attr_max_fail_p = [max(link_attr_max_fail_p[j], path_fail_softmax[j]) for j in range(self.m)]

        return link_attr_max_fail_p

    def get_link_attr(self, env_actions):
        link_capacity = self.get_link_capacity()
        link_capacity_available = self.get_link_capacity_available(env_actions)
        link_betweenness = self.jieshu_ret    # link介数
        
        link_attr = [[link_capacity[i], link_capacity_available[i], link_betweenness[i]] for i in range(self.m)]
        return link_attr

    def get_path_attr(self):
        return [[float(flow.bw)] for flow in self.flows]
    
    def get_mask(self, now_actions):
        mask = [[False for _ in range(self.m)] for _ in range(self.f)]
        for i in range(self.f):
            flow = self.flows[i]
            path = flow.paths[now_actions[i]]
            for j in range(len(path) - 1):
                link_id = self.get_edgeId_by_node(path[j], path[j + 1])
                mask[i][link_id] = True
        return mask

    def get_features_dfs(self, 
                         cur, 
                         env_actions, fail_flows, 
                         link_attr_list, path_attr_list, mask_list, 
                         new_actions_list, 
                         link_attr_max_fail_p_list, 
                         tensor_flag=True):
        if cur == len(fail_flows):
            link_attr = self.get_link_attr(env_actions)
            for i in range(self.m):
                link_attr[i].append(link_attr_max_fail_p_list[i])
                if link_attr[i][1] < 0:
                    return
            if tensor_flag:
                link_attr_list.append(torch.tensor(link_attr))
                path_attr_list.append(torch.tensor(self.get_path_attr()))
                mask_list.append(torch.tensor(self.get_mask(env_actions)))
            else:
                link_attr_list.append(link_attr)
                path_attr_list.append(self.get_path_attr())
                mask_list.append(self.get_mask(env_actions))
            new_actions_list.append(env_actions)
            return
        now_judge_flow_id = fail_flows[cur]
        flow = self.flows[now_judge_flow_id]
        paths_len = min(self.K_SP_CNT, len(flow.paths))
        for i in range(paths_len):
            env_actions[now_judge_flow_id] = i
            self.get_features_dfs(cur + 1, env_actions, fail_flows, link_attr_list, path_attr_list, mask_list, new_actions_list, link_attr_max_fail_p_list, tensor_flag=tensor_flag)

    def get_features(self, env_actions, fail_flows, device):  # env_actions: 当前环境的路径方案, fail_flows: 失效的路径id列表
        link_attr_list = []
        path_attr_list = []
        mask_list = []
        new_actions_list = []

        env_actions_copy = copy.deepcopy(env_actions)
        link_attr_max_fail_p_list = self.get_link_attr_max_fail_p_list(env_actions, fail_flows)

        self.get_features_dfs(0, env_actions_copy, fail_flows, link_attr_list, path_attr_list, mask_list, new_actions_list, link_attr_max_fail_p_list)
        # [batch_size, num_link, 4] [总带宽, 可用带宽, 介数, fail_p] fail_p量化了该边可能失效的可能性
        # [batch_size, num_path, 1] [带宽]
        # [batch_size, num_path, num_link]
        return torch.stack(link_attr_list).to(device), torch.stack(path_attr_list).to(device), torch.stack(mask_list).to(device), new_actions_list
    
    def get_fail_flows(self, env_actions, fail_links):
        fail_flows = []
        is_link_fail = [False for _ in range(self.m)]
        for link_id in fail_links:
            is_link_fail[link_id] = True
        link_cap_avi = self.get_link_capacity_available(env_actions)
        for link_id in range(self.m):
            if link_cap_avi[link_id] < 0:
                is_link_fail[link_id] = True
        for i in range(self.f):
            path = self.flows[i].paths[env_actions[i]]
            for j in range(len(path) - 1):
                link_id = self.get_edgeId_by_node(path[j], path[j + 1])
                if is_link_fail[link_id]:
                    fail_flows.append(i)
                    break
        return fail_flows
    
    def get_features_tuple_one(self, fail_links, env_actions):
        fail_flows = self.get_fail_flows(env_actions, fail_links)
        link_attr_max_fail_p_list = self.get_link_attr_max_fail_p_list(env_actions, fail_flows)
        link_attr = self.get_link_attr(env_actions)
        for i in range(self.m):
            link_attr[i].append(link_attr_max_fail_p_list[i])
        path_attr = self.get_path_attr()
        mask = self.get_mask(env_actions)
        return link_attr, path_attr, mask

    def get_features_tuple(self, fail_links_tuple, env_actions_tuple, device):
        link_attr_list = []
        path_attr_list = []
        mask_list = []

        for id in range(len(fail_links_tuple)):
            fail_links = fail_links_tuple[id]
            env_actions = env_actions_tuple[id]
            link_attr, path_attr, mask = self.get_features_tuple_one(fail_links, env_actions)

            link_attr_list.append(torch.tensor(link_attr))
            path_attr_list.append(torch.tensor(path_attr))
            mask_list.append(torch.tensor(mask))

        # [batch_size, num_link, 4] [总带宽, 可用带宽, 介数, fail_p] fail_p量化了该边可能失效的可能性
        # [batch_size, num_path, 1] [带宽]
        # [batch_size, num_path, num_link]
        return torch.stack(link_attr_list).to(device), torch.stack(path_attr_list).to(device), torch.stack(mask_list).to(device)

    def get_features_target_exprience(self, env_actions, fail_flows):   # make_exprience的target_exprience专用
        link_attr_list = []
        path_attr_list = []
        mask_list = []
        new_actions_list = []

        env_actions_copy = copy.deepcopy(env_actions)
        link_attr_max_fail_p_list = self.get_link_attr_max_fail_p_list(env_actions, fail_flows)

        self.get_features_dfs(0, env_actions_copy, fail_flows, link_attr_list, path_attr_list, mask_list, new_actions_list, link_attr_max_fail_p_list, tensor_flag=False)
        return link_attr_list, path_attr_list, mask_list
    