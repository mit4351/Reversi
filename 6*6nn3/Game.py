##!/usr/bin/python
# -*- coding: utf-8 -*-
import ReverseCommon

""" オセロゲーム """
class Game:
    def __init__(self, player1, player2, reverse_board):
        self._player1 = player1
        self._player2 = player2
        self._reverse_board = reverse_board

    def play(self, output_board):
        if output_board:
            ReverseCommon.print_board(self._reverse_board)

        # 勝負
        while True:
            # 置けなくなったら終了
            if self._reverse_board.is_game_set():
                break

            # プレーヤ1
            if self._reverse_board.is_my_turn(self._player1.color):
                next_move = self._player1.next_move(self._reverse_board)
                self._reverse_board.put_stone(self._player1.color, next_move[0], next_move[1])
                if output_board:
                    ReverseCommon.print_board(self._reverse_board)

            # 置けなくなったら終了
            if self._reverse_board.is_game_set():
                break

            # プレーヤ2
            if self._reverse_board.is_my_turn(self._player2.color):
                #next_move = self._player2.next_move(self._reverse_board)
                try:
                    next_move = self._player2.next_move(self._reverse_board)
                except ValueError as e:
                    #print(e)
                    return 36, 0

                self._reverse_board.put_stone(self._player2.color, next_move[0], next_move[1])
                if output_board:
                    ReverseCommon.print_board(self._reverse_board)

        return self.get_score()

    def get_score(self):
        player1_score = ReverseCommon.get_score(self._reverse_board.stone_status, self._player1.color)
        player2_score = ReverseCommon.get_score(self._reverse_board.stone_status, self._player2.color)
        return player1_score, player2_score

    def get_winner(self):
        # 勝者を返す
        player1_score, player2_score = self.get_score()

        if player1_score > player2_score:
            return self._player1
        else:
            return self._player2
