#!/usr/bin/python
# -*- coding: utf-8 -*-

import copy
import random
import ReverseCommon
import ReverseBoard
import Game
import sys, os
import pickle
import numpy as np
from functions import sigmoid, softmax, dtanh, init_KeysCount, myMax, myMin, myRandom, dMean_squared_error

""" プレーヤの基盤クラス(AIも含む) """
class Player:
    def __init__(self, color):
        """ コンストラクタ """
        self._color = color
        self.network = {}
        self._history = []

    def next_move(self, board):
        """ 次の手を返す """
        pass

    def save_network(self, file_name):
        pass

    def update_network(self, ritu=1, weight_init_std = 0.01):
        pass

    @property
    def color(self):
        """ 自分の色を返す """
        return self._color

""" ランダムで石を置くAI """
class RandomAi(Player):
    def next_move(self, board):
        # 石を置ける全候補地
        all_candidates = ReverseCommon.get_puttable_points(board.stone_status, self._color)
        # ランダムで次の手を選ぶ
        index = random.randint(0, len(all_candidates) - 1)
        return all_candidates[index]

"""今回の１手で最も石が取れる場所に置くAI"""
class NextStoneMaxAi(Player):
    def next_move(self, board):
        # 石を置ける全候補地
        all_candidatess = ReverseCommon.get_puttable_points(board.stone_status, self._color)
        # 今回の一手で最も石が取れる場所一覧
        filtered_candidates = []
        max_score = -1
        for candidates in all_candidatess:
            next_board = ReverseCommon.put_stone(board.stone_status, self._color, candidates[0], candidates[1])
            score = ReverseCommon.get_score(board.stone_status, self._color)
            if score >= max_score:
                filtered_candidates.append(candidates)
                max_score = score

        return filtered_candidates[random.randint(0, len(filtered_candidates) - 1)]

"""人間です。"""
class Human(Player):
    def next_move(self, board):
        all_candidates = ReverseCommon.get_puttable_points(board.stone_status, self._color)
        while True:
            try:
                # x,yの形式で入力する
                next_move_str = raw_input("next_move > ")
                next_move_str_split = next_move_str.split(",")
                if len(next_move_str_split) == 2:
                    next_move = [int(next_move_str_split[0]), int(next_move_str_split[1])]
                    if next_move in all_candidates:
                        return next_move
                    else:
                        print ("can't put there.")
            except ValueError:
                print ("format error.")

""" 最低限の手の良し悪しを知っているAI """
class RandomAiKnowGoodMove(Player):
    def next_move(self, board):
        known_good_moves = [[0, 0], [0, 7], [7, 0], [7, 7]]
        known_bad_moves = [[0, 1], [1, 0], [1, 1], [0, 6], [1, 6], [1, 7], [6, 0], [6, 1], [7, 1], [7, 6], [6, 7], [6, 6]]
        all_candidates = ReverseCommon.get_puttable_points(board.stone_status, self._color)
        # 4隅が取れるなら取る
        good_moves = list(filter(lambda good_move: good_move in known_good_moves, all_candidates))

        if len(good_moves) > 0:
            return good_moves[random.randint(0, len(good_moves) - 1)]

        # 4隅に隣接する場所は避ける
        #not_bad_moves = filter(lambda  not_bad_move: not_bad_move not in (known_good_moves + known_bad_moves), all_candidates)
        not_bad_moves = list(filter(lambda  not_bad_move: not_bad_move not in (known_good_moves + known_bad_moves), all_candidates))
        if len(not_bad_moves) > 0:
            return not_bad_moves[random.randint(0, len(not_bad_moves) - 1)]

        return all_candidates[random.randint(0, len(all_candidates) - 1)]


""" DQNで石を置くAI (Deep Q-Network)"""
class DeepQNetWork(Player):
    def __init__(self, color, filename):
        super(DeepQNetWork, self).__init__(color)
        self.KeysCount = {}
        self.load_network(filename)

    def load_network(self, file_name="DQN_BLACK.pkl"):
        if os.path.isfile(file_name) :
           with open(file_name, 'rb') as f:
               obj = pickle.load(f)
               self.KeysCount  =   obj[0]
               self.network    =   obj[1]
        else:
           self.KeysCount = init_KeysCount(8, 0.5, 5)

        count = self.network['SC']
        print("累積学習=%s" %(count))

        #print(sorted(self.KeysCount.items(), key=lambda x: x[0]))
        for obj in sorted(self.KeysCount.items(), key=lambda x: int(x[0])):
            print(str(obj[0]) + "=" + str(obj))

    def save_network(self, file_name="DQN_BLACK.pkl"):
        with open(file_name, 'wb') as f:
             pickle.dump([self.KeysCount, self.network], f)

    def update_network(self, ritu=1, weight_init_std = 0.001):
        #print('*'*10 + '(update_network)' + '*'*10)
        #print(str(select) + "::" + str(ReverseCommon.IndexToPoint(select)))
        #self._history.append([Key,select])
        for i in range(0,len(self._history)):
            key = self._history[i][0]
            act = self._history[i][1]
            #print(str(i+1) + " key=" + key + " act=" + str(act) + " qWait=" + str(self.network[key][act]))
            if(key in self.network):
                acc = self.network[key][act] + ritu * weight_init_std
                if acc > 10:
                    acc = 10
                elif acc < -10:
                    acc = -10
                self.network[key][act] = acc
            #self.network[key][act] += ((i % 22) + 1)  *  ritu * weight_init_std
            #print(self.network[key])
        self._history = []

    def next_move(self, board):
        #Key = ReverseCommon.boardToKey(board.stone_status)
        Key = ReverseCommon.boardToKey3(board.stone_status)
        #print(Key)
        actions = ReverseCommon.get_puttable_points(board.stone_status, self._color)
        #print(actions)
        #print(Key in self.network)
        if(not (Key in self.network)):
            self.wait_set(Key, actions, board)

        #print(self.network)
        #print(self.network[Key])
        #print(len(self._history))
        if (Key in self.network):
            if self._color == ReverseCommon.BLACK:
                select = min(self.network[Key],key=(lambda x:self.network[Key][x]))
            else:
                select = max(self.network[Key],key=(lambda x:self.network[Key][x]))
        else:
            # 石を置ける全候補地からランダムで次の手を選ぶ
            #all_candidates = ReverseCommon.get_puttable_points(board.stone_status, self._color)
            #index = random.randint(0, len(all_candidates) - 1)
            #select = ReverseCommon.PointToIndex(all_candidates[index])

            # 石を置ける全候補地
            all_candidatess = ReverseCommon.get_puttable_points(board.stone_status, self._color)
            # 今回の一手で最も石が取れる場所一覧
            filtered_candidates = []
            max_score = -1
            for candidates in all_candidatess:
                score = ReverseCommon.get_score(board.stone_status, self._color)
                if score >= max_score:
                    filtered_candidates.append(candidates)
                    max_score = score

            index = random.randint(0, len(filtered_candidates) - 1)
            select = ReverseCommon.PointToIndex(filtered_candidates[index])
            #filtered_candidates[random.randint(0, len(filtered_candidates) - 1)]

        #print(str(select) + "::" + str(ReverseCommon.IndexToPoint(select)))
        self._history.append([Key,select])
        return ReverseCommon.IndexToPoint(select)

    """
    Dropoutを実装する
    この学習モデルは(白)後手番学習
    Keyは手数によって(2手目6石 6**2、4手目8石 8**2、・・・・、60手目64石 64**2)
    通の組み合わせがある
    dropout_rait=0.5仮
    各手番の組み合わせ数 * dropout_rait で調整を試みる
    """
    def wait_set(self, Key, actions, board):

        aa = ReverseCommon.get_remain(board.stone_status)
        bb = self.KeysCount[str(64-aa)]
        #print("%2d::%s::%2d::%2d::%2d::%2d" % (len(self._history),Key,aa,64-aa,bb[0],bb[1]))

        #if (len(self._history) < 2): return
        if (bb[1] >= bb[0]): return
        #print(Key)
        self.KeysCount[str(64-aa)][1] += 1
        wait = {}
        Waits = np.random.random(len(actions))
        i = 0
        for wact in  actions:
            wPoint = ReverseCommon.PointToIndex(wact)
            wait[wPoint] = Waits[i]
            i += 1
            #print(str(wact) + ":::" + str(wPoint) + ":::"  + str(wait[wPoint]))

        self.network[Key] = wait
        #print(self.network[Key] )

class QNetWork(Player):
    def __init__(self, color, filename):
        super(QNetWork, self).__init__(color)
        self.load_network(filename)

    def load_network(self,file_name):
        if os.path.isfile(file_name) :
           with open(file_name, 'rb') as f:
               self.network    =  pickle.load(f)

    def save_network(self, file_name):
        with open(file_name, 'wb') as f:
             pickle.dump(self.network, f)

    def update_network(self, ritu=1, weight_init_std = 0.001):
        ALPHA = 0.1
        GAMMA = 0.99
        for  i in range(0,len(self._history)-1):
            Key, select = self._history[i][0], self._history[i][1]
            #self.network[Key][select] += ALPHA * (reward + GAMMA*self.network[next_s].max() - self.network[Key][select])
            next_key =self._history[i+1][0]
            if next_key in self.network:
                a = min(self.network[next_key],key=(lambda x:self.network[next_key][x]))
                aaa = self.network[next_key][a]
                #print(self.network[next_key][a])
            else:
                aaa = 0.5
            #print("Key=%s::select=%s" %(Key,select))
            bbb = 0.0
            if Key in self.network:
                bbb = self.network[Key][select]
                self.network[Key][select] += ALPHA * (ritu + GAMMA*aaa - bbb)
            #print("bef=%f: aft=%f" %(mmm,self.network[Key][select]))
            #print("bbb=%s::aaa=%s" %(bbb,aaa))
        self._history = []

    def wait_set(self, Key, actions):
        #print(Key)
        if len(actions) < 2:
            return
        wait = {}
        Waits = np.random.random(len(actions))
        i = 0
        for wact in  actions:
            wPoint = ReverseCommon.PointToIndex(wact)
            wait[wPoint] = Waits[i]

        self.network[Key] = wait

    def next_move(self, board):
        Key = ReverseCommon.boardToKey3(board.stone_status)
        #print(Key)
        # 石を置ける全候補地
        actions = ReverseCommon.get_puttable_points(board.stone_status, self._color)

        if(not (Key in self.network)):
            self.wait_set(Key, actions)

        if (Key in self.network):
            if self._color == ReverseCommon.BLACK:
                select = min(self.network[Key],key=(lambda x:self.network[Key][x]))
            else:
                select = max(self.network[Key],key=(lambda x:self.network[Key][x]))
            #print("select=%s" % (select))
            result = ReverseCommon.IndexToPoint(select)
        else:
            # ランダムで次の手を選ぶ
            index = random.randint(0, len(actions) - 1)
            result = actions[index]
            select = ReverseCommon.PointToIndex(result)

        self._history.append([Key,select])
        return result
