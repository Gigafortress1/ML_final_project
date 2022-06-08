import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import networkx as nx

from tqdm import tqdm

import json

import pandas as pd


SCALE = 25
print('!!! SCALE = {} !!!'.format(SCALE))

if SCALE == 10:
    NR = 4172
    NV = 3151
    START_P = 206

elif SCALE == 25:
    NR = 486
    NV = 382
    START_P = 104

else:
    NR = np.inf
    NV = np.inf
    START_P = 0

MAP_SIZE = (800, 600)

COLOR_POI = ['red', 'orange', 'yellow', 'lime', 'cyan', 'deepskyblue', 'blue', 'purple']


DIR_DATA = '../data'
DIR_ISF = DIR_DATA + '/{}'.format(SCALE)
DIR_F = DIR_ISF + '/final'

F_ROADS_INFO = DIR_F + '/roads_info.csv'
F_SPD_NET = DIR_F + '/spd_net.csv'
F_R2VV = DIR_F + '/r2vv.json'
F_V2R = DIR_F + '/v2r.json'
F_V2VS = DIR_F + '/v2vs.json'
F_V2GPS = DIR_F + '/v2gps.json'
F_V2POI = DIR_F + '/v2poi.json'
F_MAP = DIR_F + '/map.json'
F_POI = DIR_F + '/poi.json'
F_POI_COUNT = DIR_F + '/poi_count.json'
F_POI_MOST = DIR_F + '/poi_most.json'
F_DES_P = DIR_F + '/des_p.json'


def json_load(pth, if_str_to_int=True):
    with open(pth, 'r') as f:
        res = json.load(f)
    if if_str_to_int:
        res = dict_key_str_to_int(res)
    return res


def dict_key_str_to_int(d):
    return {int(k): v for k, v in d.items()}


def dict_max(d):
    dl = [(k, v) for k, v in d.items()]
    sort_d = sorted(dl, key=lambda x: -x[1])
    return sort_d[0]


def r2vvTvv2r(r2vv):
    vv2r = {}
    for r, vv in r2vv.items():
        vv2r[(vv[0], vv[1])] = r
        vv2r[(vv[1], vv[0])] = r
    return vv2r


def draw_map():
    roads_info = pd.read_csv(F_ROADS_INFO, header=0)

    v2gps = json_load(F_MAP)

    with open(F_R2VV, 'r') as f:
        r2vv = json.load(f)
    es = [tuple(v) for v in r2vv.values()]

    with open(F_V2R, 'r') as f:
        v2r = json.load(f)

    ws = list(roads_info['width'])
    std_w = np.std(ws)
    max_w = np.max(ws)
    ws = [w / std_w for w in ws]

    with open(F_DES_P, 'r') as f:
        des_p = json.load(f)
    ns = list(range(NV))
    colors_v = ['cornflowerblue' for _ in range(NV)]
    size_v = [10 for _ in range(NV)]
    for tp, nl in des_p.items():
        clr = COLOR_POI[int(tp)]
        for n in nl:
            ns.append(n)
            colors_v.append(clr)
            size_v.append(50)
    ns.append(START_P)
    colors_v.append('black')
    size_v.append(100)

    vs = set()
    for vv in r2vv.values():
        for v in vv:
            vs.add(v)

    net = nx.Graph()
    # print(net.nodes)
    nx.draw_networkx_nodes(net, v2gps, nodelist=ns, node_size=size_v, node_color=colors_v, alpha=0.5)
    # nx.draw_networkx_nodes(net, v2gps, nodelist=ns, node_size=100, node_color='cornflowerblue', alpha=0.1)
    nx.draw_networkx_edges(net, v2gps, edgelist=es, width=1, edge_color='dimgrey')
    # nx.draw_networkx_edges(net, v2gps, edgelist=es, width=ws, edge_color=edge_clr)

    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Beijing Roads Network and POI')
    s = 1
    plt.xlim(0, MAP_SIZE[0])
    plt.ylim(0, MAP_SIZE[1])

    # my_save_fig(D_FIG + '/road_network_poi.png')

    plt.show()


Episode = 100
MAX_ITER = 100 * SCALE


class QLearningAgent:
    def __init__(self, learning_rate=0.001, reward_decay=0.9, epsilon=0.1):
        self.lr = learning_rate
        self.discount_factor = reward_decay
        self.epsilon = epsilon

        self.vv2r = r2vvTvv2r(json_load(F_R2VV))
        self.v2vs = json_load(F_V2VS)
        self.v2poi = json_load(F_V2POI)
        self.road_info = pd.read_csv(F_ROADS_INFO, header=0)

        self.q_table = {}
        for v1, vs in self.v2vs.items():
            self.q_table[v1] = {}
            for v2 in vs:
                self.q_table[v1][v2] = 0

        self.poi_if_reach = [0 for _ in range(8)]
        self.reach_n = 0
        # self.now_at = START_P

    def learn(self, state, next_state, reward):
        current_q = self.q_table[state][next_state]
        # 贝尔曼方程更新
        new_q = reward + self.discount_factor * dict_max(self.q_table[next_state])[1]
        self.q_table[state][next_state] += self.lr * (new_q - current_q)

    def get_next_state(self, state):
        return self.v2vs[state]

    def choose_next_state(self, former_state, state):
        next_states = self.get_next_state(state)
        if np.random.rand() < self.epsilon:
            next_state = np.random.choice(next_states)
        else:
            next_state = dict_max(self.q_table[state])[0]
            if next_state == former_state:
                next_state = np.random.choice(next_states)
        return next_state

    def get_observation(self, former_state, state, next_state):
        reward = 0
        try:
            tps = self.v2poi[next_state]
            for tp in tps:
                if self.poi_if_reach[tp] == 0:
                    self.poi_if_reach[tp] = 1
                    self.reach_n += 1
                    reward += 100 * self.reach_n
        except KeyError:
            pass

        if len(self.v2vs[next_state]) == 1:
            reward -= 10

        # if next_state == former_state:
        #     reward -= 5

        # r = self.vv2r[(state, next_state)]
        # length = self.road_info.at[r, 'length']
        # reward += 1 / length

        if self.reach_n == 8:
            if_done = True
        else:
            if_done = False

        return reward, if_done

    def reset(self):
        self.poi_if_reach = [0 for _ in range(8)]
        self.reach_n = 0


class Env(object):
    def __init__(self, root):

        # 创建画布
        self.root = root
        self.width = MAP_SIZE[0]
        self.height = MAP_SIZE[1]
        print(self.width)
        print(self.height)

        # 城市数目初始化为city_num
        self.n = NV
        self.v2gps = json_load(F_MAP)
        self.r2vv = json_load(F_R2VV)
        self.v2poi = json_load(F_V2POI)
        self.vs = []

        # tkinter.Canvas
        self.canvas = tk.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="#EBEBEB",  # 背景白色
            xscrollincrement=1,
            yscrollincrement=1
        )
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        # self.canvas.pack()
        self.title("QLearning(n:初始化 e:开始搜索 s:停止搜索 q:退出程序)")
        self.draw_map()

    def title(self, s):
        self.root.title(s)

    # def __bindEvents(self):
    #
    #     self.root.bind("q", self.quite)  # 退出程序
    #     self.root.bind("n", self.new)  # 初始化
    #     self.root.bind("e", self.search_path)  # 开始搜索
    #     self.root.bind("s", self.stop)  # 停止搜索

    def draw_map(self, evt=None):
        for v, gps in self.v2gps.items():
            x, y = gps
            r = 4
            try:
                r += len(self.v2poi[v]) * 3
                color = COLOR_POI[self.v2poi[v][-1]]
            except KeyError:
                color = '#00BFFF'

            node = self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                           fill=color,  # DeepSkyBlue
                                           outline="white",  # 轮廓白色
                                           tags="node",
                                           )
            self.vs.append(node)

        for vv in self.r2vv.values():
            self.canvas.create_line(self.v2gps[vv[0]], self.v2gps[vv[1]], fill="#000000", tags="line")

    def draw_path(self, vs):
        for v in vs:
            x, y = self.v2gps[v]
            self.canvas.create_oval(x - 5,
                                    y - 5, x + 5, y + 5,
                                    fill='#FF0000',  # DeepSkyBlue
                                    outline="white",  # 轮廓白色
                                    tags="node",
                                    )
            self.canvas.update()

    def mainloop(self):
        self.root.mainloop()

    # def draw_path(self, ):


if __name__ == '__main__':
    draw_map()
    quit()

    ql = QLearningAgent()
    env = Env(tk.Tk())

    max_iter = MAX_ITER
    best_path = []
    for i in tqdm(range(Episode)):
        path = []
        fst = -1
        st = START_P
        path.append(st)
        for j in range(max_iter):
            nst = ql.choose_next_state(fst, st)
            rwd, if_d = ql.get_observation(fst, st, nst)
            ql.learn(st, nst, rwd)
            fst = st
            st = nst
            path.append(st)
            if if_d:
                print('SUCCESS!')
                print(path)
                if j + 1 < max_iter:
                    max_iter = j + 1
                    best_path = path
                env.draw_path(path)

    env.draw_path(path)
    # if len(best_path) == 0:
    #     best_path = path
    # print(best_path)























