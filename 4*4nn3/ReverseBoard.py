#!/usr/bin/python
# -*- coding: utf-8 -*-
import ReverseCommon

""" オセロ盤 """
class ReverseBoard:
    def __init__(self):
        """ Constructor """
        # ボード初期化
        self._stone_status = [[ReverseCommon.NONE for i in range(4)] for j in range(4)]
        self._stone_status[1][1] = ReverseCommon.WHITE
        self._stone_status[2][2] = ReverseCommon.WHITE
        self._stone_status[1][2] = ReverseCommon.BLACK
        self._stone_status[2][1] = ReverseCommon.BLACK
        # 黒のターンに初期化
        self._turn = ReverseCommon.BLACK

    """ 交代 """
    def change_turn(self):
        if self._turn == ReverseCommon.WHITE:
            self._turn = ReverseCommon.BLACK
        else:
            self._turn = ReverseCommon.WHITE

    """ 置く & ひっくり返す """
    def put_stone(self, color, i, j):
        self._stone_status = ReverseCommon.put_stone(self._stone_status, color, i, j)

        # プレーヤ交代
        enemy = not(color)
        if len(ReverseCommon.get_puttable_points(self._stone_status, enemy)) > 0:
            self.change_turn()

    """ 置く & ひっくり返す """
    def put_stone2(self, next_move):
        self.put_stone(self._turn, next_move[0], next_move[1])

    """ ゲームセットか返す　"""
    def is_game_set(self):
        return ReverseCommon.is_game_set(self._stone_status)

    """ 自分のターンか返す """
    def is_my_turn(self, color):
        if self._turn == color:
            return True
        return False

    @property
    def CurrentColor(self):
        return self._turn

    @property
    def stone_status(self):
        """ 盤面を返す"""
        return self._stone_status

class CustomReverseBoard(ReverseBoard):
    """ 途中状態の盤面を作るようのクラス """
    def __init__(self, board, turn):
        self._stone_status = board
        self._turn = turn
