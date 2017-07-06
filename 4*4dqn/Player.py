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
        known_good_moves = [[0, 0], [0, 3], [3, 0], [3, 3]]
        all_candidates = ReverseCommon.get_puttable_points(board.stone_status, self._color)
        # 4隅が取れるなら取る
        good_moves = list(filter(lambda good_move: good_move in known_good_moves, all_candidates))

        if len(good_moves) > 0:
            return good_moves[random.randint(0, len(good_moves) - 1)]

        return all_candidates[random.randint(0, len(all_candidates) - 1)]

class QNetWork(Player):
    def __init__(self, color, filename, trFlg=False):
        super(QNetWork, self).__init__(color)
        self.load_network(filename)
        self.trFlg = trFlg

    def load_network(self,file_name):
        if os.path.isfile(file_name) :
           with open(file_name, 'rb') as f:
               self.network    =  pickle.load(f)
        else:
           self.network['SC']  =   0

        count = self.network['SC']
        print("累積学習=%s" %(count))

    def save_network(self, file_name):
        with open(file_name, 'wb') as f:
             pickle.dump(self.network, f)

    def update_network(self, pPoint=0, weight_init_std = 0.001):
        count = self.network['SC']
        count += 1
        ALPHA = 0.1
        GAMMA = 0.99
        ritu = 0
        if pPoint > 0:
            ritu = 1
        elif pPoint == 0:
            ritu = 0
        else:
            ritu = -0.1

        for  i in range(0,len(self._history)-1):
            Key, select = self._history[i][0], self._history[i][1]
            #self.network[Key][select] += ALPHA * (reward + GAMMA*self.network[next_s].max() - self.network[Key][select])
            next_key =self._history[i+1][0]
            if next_key in self.network:
                next_a = max(self.network[next_key],key=(lambda x:self.network[next_key][x]))
                nex_max = self.network[next_key][next_a]
            else:
                nex_max = 0.1

            #print("Key=%s::select=%s::ritu=%s" %(Key, select, ritu))

            if Key in self.network:
                befor = self.network[Key][select]
                #print("Key=%s::select=%s::ritu=%s" %(Key, select, ritu))
                self.network[Key][select] += ALPHA * (ritu + GAMMA*nex_max - befor)

        self._history = []
        self.network['SC'] = count
        return count

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
        EPS = 0.1
        Key = ReverseCommon.boardToKey3(board.stone_status)
        #print(Key)
        # 石を置ける全候補地
        actions = ReverseCommon.get_puttable_points(board.stone_status, self._color)

        if(not (Key in self.network)):
            self.wait_set(Key, actions)

        if self.trFlg and np.random.random() < EPS:
            # ランダムで次の手を選ぶ
            index = random.randint(0, len(actions) - 1)
            result = actions[index]
            select = ReverseCommon.PointToIndex(result)
        elif (Key in self.network):
            if self._color == ReverseCommon.BLACK:
                select = min(self.network[Key],key=(lambda x:self.network[Key][x]))
            else:
                select = max(self.network[Key],key=(lambda x:self.network[Key][x]))
            #print("select=%s" % (select))
            result = ReverseCommon.IndexToPoint(select)
        else:
            result = self.maxstone(board)
            select = ReverseCommon.PointToIndex(result)

        self._history.append([Key,select])
        return result

    def maxstone(self, board):
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
