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
        self.network2 = {}
        self._history = []
        self._history2 = []

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


class QNetWork(Player):
    def __init__(self, color, filename, filename2):
        super(QNetWork, self).__init__(color)
        self.load_network(filename,filename2)

    def load_network(self,file_name,file_name2):
        if os.path.isfile(file_name) :
           with open(file_name, 'rb') as f:
               self.network    =  pickle.load(f)
        if os.path.isfile(file_name2) :
           with open(file_name2, 'rb') as f:
               self.network2    =  pickle.load(f)
        else:
           self.network2  =   self.init_wait()

        count = self.network2['SC']
        print("累積学習=%s" %(count))

    def init_wait(self):
        n_in = 64  #8*8 箇所
        n_hiddn1 = 100 #300  #100
        n_hiddn2 = 200
        n_hiddn3 = 200
        n_out = 65  #8*8 箇所 + 1(パス)

        result = {}
        result['SC'] = 0  #累積学習
        result['W1'] = np.random.uniform(low=-0.1, high=0.1, size=(n_in, n_hiddn1))
        result['b1'] = np.random.uniform(low=-0.1 , high=0.1, size=(n_hiddn1))
        result['W2'] = np.random.uniform(low=-0.1, high=0.1, size=(n_hiddn1, n_hiddn2))
        result['b2'] = np.random.uniform(low=-0.1 , high=0.1, size=(n_hiddn2))
        result['W3'] = np.random.uniform(low=-0.1, high=0.1, size=(n_hiddn2, n_hiddn3))
        result['b3'] = np.random.uniform(low=-0.1 , high=0.1, size=(n_hiddn3))
        result['W4'] = np.random.uniform(low=-0.1, high=0.1, size=(n_hiddn3, n_out))
        result['b4'] = np.random.uniform(low=-0.1 , high=0.1, size=(n_out))
        return result

    def save_network(self, file_name, file_name2):
        with open(file_name, 'wb') as f:
             pickle.dump(self.network, f)
        with open(file_name2, 'wb') as f:
             pickle.dump(self.network2, f)

    def update_network(self, pPoint=0, weight_init_std = 0.001):
        count = self.network2['SC']
        count += 1
        ALPHA = 0.1
        GAMMA = 0.99
        ritu = 0
        if pPoint > 0:
            ritu = 1
        else:
            ritu = -0.5

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
                #print("bef=%f: aft=%f" %(befor,self.network[Key][select]))
                #print("bef=%f: aft=%f" %(mmm,self.network[Key][select]))

        for i in range(0,len(self._history2)):
            x  = self._history2[i][0]
            y  = self._history2[i][1]
            z1 = self._history2[i][2]
            z2 = self._history2[i][3]
            z3 = self._history2[i][4]
            p  = self._history2[i][5]
            m3  = self._history2[i][6]
            msk  = self._history2[i][7]
            all_candidates  = self._history2[i][8]
            self.back_propagation(x, z1, z2, z3, y, p, m3, msk, all_candidates, i+1, pPoint)
        #print(self._history[0][1])


        self._history = []
        self._history2 = []
        self.network2['SC'] = count
        return count

    def wait_set(self, Key, actions):
        if len(actions) < 2:
            return
        wait = {}
        Waits = softmax(np.random.random(len(actions)))
        #for wact in  actions:
        for i in range(0,len(actions)):
            wPoint = ReverseCommon.PointToIndex(actions[i])
            wait[wPoint] = Waits[i]

        self.network[Key] = wait

    def next_move(self, board):
        #print(Key)
        # 石を置ける全候補地
        all_candidates = ReverseCommon.get_puttable_points(board.stone_status, self._color)

        known_good_moves = [[0, 0], [0, 7], [7, 0], [7, 7]]
        # 4隅が取れるなら取る
        good_moves = list(filter(lambda good_move: good_move in known_good_moves, all_candidates))
        epc = random.randint(1,30)
        if epc > 15 and len(good_moves) > 0:
            Key = ReverseCommon.boardToKey3(board.stone_status)
            result = good_moves[random.randint(0, len(good_moves) - 1)]
            select = ReverseCommon.PointToIndex(result)
            self._history.append([Key,select])
        elif(len(self._history2) > 23):
            Key = ReverseCommon.boardToKey3(board.stone_status)
            if  epc > 15:
                if(not (Key in self.network)):
                    self.wait_set(Key, all_candidates)

            if(Key in self.network):
                if self._color == ReverseCommon.BLACK:
                    select = min(self.network[Key],key=(lambda x:self.network[Key][x]))
                else:
                    select = max(self.network[Key],key=(lambda x:self.network[Key][x]))
            else:
                #index = random.randint(0, len(all_candidates) - 1)
                #result = all_candidates[index]
                result = self.maxstone(board)
                select = ReverseCommon.PointToIndex(result)
                #print("select=%s" % (select))
            result = ReverseCommon.IndexToPoint(select)
            self._history.append([Key,select])
        else:
            x = ReverseCommon.boardToKey2(board.stone_status)
            msk = ReverseCommon.PointsToMask(all_candidates)
            y, z1, z2, z3, m3 = self.predict(x)
            if self._color == ReverseCommon.BLACK:
                p = myMin(y * msk)
            else:
                p = myMax(y * msk)

            result = ReverseCommon.IndexToPoint(p)

            self._history2.append([x, y, z1, z2, z3, p, m3, msk, all_candidates])
            #print("wPoint=%s :: all_candidates=%s :: p=%s  ::  y=%s"  % (wPoint, all_candidates, p, y))
            if not result in all_candidates:
                print('-*-'*10 + " デバッグ"  +  '-*-'*10)
                print("wPoint=%s::all_candidates=%s::p=%s::y=%s"  % (result, all_candidates, p, y))
                if self._color == ReverseCommon.BLACK:
                    myMin(y, True)
                else:
                    myMax( y, True)
                print('*-*'*10 + " デバッグ"  +  '*-*'*10)

        return result

    def Jyouseki(self, all_candidates):
        #print("5 all_candidates=%s" %(all_candidates))
        known_good_moves = [[0, 0], [0, 7], [7, 0], [7, 7]]
        known_bad_moves = [[0, 1], [1, 0], [1, 1], [0, 6], [1, 6], [1, 7], [6, 0], [6, 1], [7, 1], [7, 6], [6, 7], [6, 6]]
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
        W1, W2, W3, W4 = self.network2['W1'], self.network2['W2'], self.network2['W3'], self.network2['W4']
        b1, b2, b3, b4 = self.network2['b1'], self.network2['b2'], self.network2['b3'], self.network2['b4']
        a1 = np.dot(x, W1) + b1
        z1 = np.tanh(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = np.tanh(a2)
        a3 = np.dot(z2, W3) + b3
        z3 = np.tanh(a3)
        m3 = np.random.binomial(1, dropout, size=a3.shape)
        z3 *=m3    # Dropout in layer 3

        a4 = np.dot(z3, W4) + b4
        y = softmax(a4)
        #y = a4
        #print("a3.shape=%s :: z3.shape=%s  :: a3=%s :: z3=%s" % (a3.shape, z3.shape, a3, z3))
        return y, z1, z2, z3, m3

    # 誤差逆伝播
    def back_propagation(self, x, z1, z2, z3, y, p, m3, msk, all_candidates, i, pPoint, learning_rate=0.001):
        W1,W2,W3,W4 = self.network2['W1'], self.network2['W2'], self.network2['W3'], self.network2['W4']
        #y = softmax(pY)
        t = np.zeros(len(y))
        if pPoint >= 0:
            t[p] = 1
        else:
            #pp = myRandom(y)
            #print("p=%s :: pp=%s" % (p,pp))
            #t[pp] = 1
            mp = self.Jyouseki(all_candidates)
            cp = ReverseCommon.PointToIndex(mp)
            t[cp] = 1

        #print("t.shape=%s p=%s t=%s" % (t.shape, p, t))
        dC_da4 = dMean_squared_error(y, t)
        aaa = dC_da4 * msk
        #aaa = dC_da4
        dC_dW4 = np.dot(z3.T.reshape(len(z3),1), aaa.reshape(1,len(aaa)))

        dC_dz3 = np.dot(aaa, W4.T)
        #dC_dz3 = np.dot(dC_da4, W4.T)
        dC_da3 = dC_dz3 * dtanh(z3) * m3
        dC_dW3 = np.dot(z2.T.reshape(len(z2),1), dC_da3.reshape(1,len(dC_da3)))

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
        W4 -= dC_dW4 * learning_rate
        self.network2['W1'], self.network2['W2'], self.network2['W3'], self.network2['W4'] = W1, W2, W3, W4
