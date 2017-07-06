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

class NNetWork(Player):
    def __init__(self, color, filename, train_flg=False):
        super(NNetWork, self).__init__(color)
        self.load_network(filename)
        self.train_flg = train_flg

    def load_network(self, file_name="NNetWork.pkl"):
        if os.path.isfile(file_name) :
           with open(file_name, 'rb') as f:
               self.network  =   pickle.load(f)
        else:
           self.network  =   self.init_wait()

        count = self.network['SC']
        print("累積学習=%s" %(count))
        self.MX = self.network['MX']
        print("累積学習進度=%s" % (self.MX))
        self.TMX = 0
        self.TMI = 100

    def init_wait(self):
        n_in = 16  #4*4 箇所
        n_hiddn1 = 100 #300  #100
        n_hiddn2 = 200
        n_out = 17  #4*4 箇所 + 1(パス)

        result = {}
        result['SC'] = 0  #累積学習
        result['MX'] = 0  #累積学習進度
        result['W1'] = np.random.uniform(low=-0.1, high=0.1, size=(n_in, n_hiddn1))
        result['b1'] = np.random.uniform(low=-0.1 , high=0.1, size=(n_hiddn1))
        result['W2'] = np.random.uniform(low=-0.1, high=0.1, size=(n_hiddn1, n_hiddn2))
        result['b2'] = np.random.uniform(low=-0.1 , high=0.1, size=(n_hiddn2))
        result['W3'] = np.random.uniform(low=-0.1, high=0.1, size=(n_hiddn2, n_out))
        result['b3'] = np.random.uniform(low=-0.1 , high=0.1, size=(n_out))
        return result

    def save_network(self, file_name="NNetWork.pkl"):
        MX = self.network['MX']
        with open(file_name, 'wb') as f:
             pickle.dump(self.network, f)

    def update_network(self, pPoint=0, learning_rate=0.001):
        count = self.network['SC']
        count += 1
        if self.TMI < len(self._history):
            Point = len(self._history)

        for i in range(0,len(self._history)):
            x  = self._history[i][0]
            y  = self._history[i][1]
            z1 = self._history[i][2]
            z2 = self._history[i][3]
            p  = self._history[i][4]
            msk  = self._history[i][5]
            wMaxPoint =  self._history[i][6]
            all_candidates  = self._history[i][7]
            self.back_propagation(x, z1, z2, y, p, msk, wMaxPoint, all_candidates, i+1, pPoint)
        #print(self._history[0][1])
        MX = self.network['MX']
        if(MX<len(self._history)):
            self.network['MX'] = len(self._history)
            print("現在学習進度=%s" %(len(self._history)))

        if self.TMX < len(self._history):
            self.TMX = len(self._history)

        if self.TMI > len(self._history):
            self.TMI = len(self._history)

        self._history = []
        self.network['SC'] = count
        return count

    def DTMX(self):
        result1 =self.TMI
        result2 =self.TMX
        self.TMI = 100
        self.TMX = 0
        return result1, result2

    def next_move(self, board):
        # 石を置ける全候補地
        all_candidates = ReverseCommon.get_puttable_points(board.stone_status, self._color)
        x = ReverseCommon.boardToKey2(board.stone_status)
        wPoint = self.brain(x, all_candidates,board)

        if not wPoint in all_candidates:
            wPoint = self.Jyouseki(all_candidates)

        return wPoint

    def brain(self,x,all_candidates, board):
        msk = ReverseCommon.PointsToMask(all_candidates)
        y, z1, z2 = self.predict(x)
        #p = np.argmax(y)
        if self._color == ReverseCommon.BLACK:
            p = myMin(y)
            #p = myMin(y * msk)
        else:
            p = myMax(y)
            #p = myMax(y * msk)

        wPoint = ReverseCommon.IndexToPoint(p)
        wMaxPoint = self.maxstone(board)

        self._history.append([x, y, z1, z2, p, msk, wMaxPoint, all_candidates])

        err_flg = False
        if not wPoint in all_candidates:
            if self.train_flg:
                err_flg = True
                raise ValueError("otetuki!")

        #print("wPoint=%s :: all_candidates=%s :: p=%s  ::  y=%s"  % (wPoint, all_candidates, p, y))

        """
        if not wPoint in all_candidates:
            print('-*-'*10 + " デバッグ"  +  '-*-'*10)
            print("wPoint=%s::all_candidates=%s::p=%s::y=%s"  % (wPoint, all_candidates, p, y))
            if self._color == ReverseCommon.BLACK:
                myMin(y, True)
            else:
                myMax( y, True)
            print('*-*'*10 + " デバッグ"  +  '*-*'*10)
        """

        #self.back_propagation(x, z1, z2, y, p, msk, 10, 1)
        return wPoint

    def Jyouseki(self, all_candidates):
        #print("5 all_candidates=%s" %(all_candidates))
        known_good_moves = [[0, 0], [0, 3], [3, 0], [3, 3]]
        # 4隅が取れるなら取る
        good_moves = list(filter(lambda good_move: good_move in known_good_moves, all_candidates))
        if len(good_moves) > 0:
            return good_moves[random.randint(0, len(good_moves) - 1)]

        return all_candidates[random.randint(0, len(all_candidates) - 1)]

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

    #順伝播
    def predict(self, x, dropout=0.15):
        W1, W2, W3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']
        a1 = np.dot(x, W1) + b1
        z1 = np.tanh(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = np.tanh(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        #y = a4
        #print("a3.shape=%s :: z3.shape=%s  :: a3=%s :: z3=%s" % (a3.shape, z3.shape, a3, z3))
        return y, z1, z2

    # 誤差逆伝播
    def back_propagation(self, x, z1, z2, y, p, msk, wMaxPoint, all_candidates, i, pPoint, learning_rate=0.001):
        W1,W2,W3 = self.network['W1'], self.network['W2'], self.network['W3']
        #y = softmax(pY)
        t = np.zeros(len(y))
        if pPoint >= 0:
            t[p] = 1
        else:
            #i = random.randint(1,30)
            '''
            if  i > 20:
                mp = wMaxPoint
            elif  i > 10:
                mp = self.Jyouseki(all_candidates)
            else:
                mp = all_candidates[random.randint(0, len(all_candidates) - 1)]
            '''
            #print("wMaxPoint=%s" %(wMaxPoint))
            #print("mp=%s" %(mp))
            mp = wMaxPoint
            cp = ReverseCommon.PointToIndex(mp)
            t[cp] = 1

        #print("t.shape=%s p=%s t=%s" % (t.shape, p, t))
        dC_da3 = dMean_squared_error(y, t)
        #aaa = dC_da3 * msk
        aaa = dC_da3
        dC_dW3 = np.dot(z2.T.reshape(len(z2),1), aaa.reshape(1,len(aaa)))

        #print("** z2.shape=%s :: dC_da3.shape=%s" % (z2.shape, dC_da3.shape))
        #print("** aaa.shape=%s :: msk.shape=%s" % (aaa.shape, msk.shape))
        dC_dz2 = np.dot(dC_da3, W3.T)
        dC_da2 = dC_dz2 * dtanh(z2)
        dC_dW2 = np.dot(z1.T.reshape(len(z1),1), dC_da2.reshape(1,len(dC_da2)))
        #print("** z1.shape=%s ::dC_da2.shape=%s" % (z1.shape, dC_da2.shape))

        dC_dz1 = np.dot(dC_da2, W2.T)
        dC_da1 = dC_dz1 * dtanh(z1)
        xx = np.array(x)
        dC_dW1 = np.dot(xx.reshape(len(xx),1), dC_da1.reshape(1,len(dC_da1)))
        #print("xx.shape=%s :: dC_da1.shape=%s" %(xx.shape,  dC_da1.shape))
        # Gradient descent update
        W1 -= dC_dW1 * learning_rate
        W2 -= dC_dW2 * learning_rate
        W3 -= dC_dW3 * learning_rate
        self.network['W1'], self.network['W2'], self.network['W3'] = W1, W2, W3
