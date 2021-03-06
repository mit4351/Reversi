#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
1.オセロの基本ロジック
2.ランダムで打つ
3.最もたくさん石が取れる手を選ぶ
4.ちょっとだけオセロの定石を知ってる
5.DQN思考エンジンを実装  <<--ここ
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
"""

import ReverseBoard
import Player
import ReverseCommon
import Game
import datetime
import sys
import MyAIGame
import matplotlib.pyplot as plt

def Training(pCount=10, pOutput=True, epoch=2):
    # 盤面を出力するか
    output = pOutput
    #print "開始(%d): %s" % (times, datetime.datetime.today())
    score_history = []

    # プレイヤー
    black_player = Player.RandomAi(ReverseCommon.BLACK)
    #black_player = Player.RandomAiKnowGoodMove(ReverseCommon.BLACK)
    #black_player = Player.NextStoneMaxAi(ReverseCommon.BLACK)

    #white_player = Player.RandomAiKnowGoodMove(ReverseCommon.WHITE)
    white_player = Player.DeepQNetWork(ReverseCommon.WHITE,"AI_Deep.pkl", train_flg=True)
    cnt = 0
    lap = 0
    tmx = 0
    tmi = 0
    for j in range(0, epoch):
        starttime    =  datetime.datetime.today()
        # 勝利数
        black_win = 0
        white_win = 0
        for i in range(0, pCount):
            # 盤面作成
            reverse_board = ReverseBoard.ReverseBoard()

            # ゲーム開始
            game = Game.Game(black_player, white_player, reverse_board)
            player1_score, player2_score = game.play(output)
            #print("player1_score=%s :: player2_score=%s " %(player1_score, player2_score))
            cnt = white_player.update_network(player2_score - player1_score)
            if i==0:
                lap = cnt
            # 勝者判定
            if player1_score >= player2_score:
                black_win += 1
            else:
                white_win += 1
            #cnt += 1

        # 勝利数
        endtime    =  datetime.datetime.today()
        result = float(white_win) / float(white_win + black_win) * 100
        tmi, tmx = white_player.DTMX()
        print ("****************** 対戦結果 epoch(%5d) **************" % j)
        print ("* 開始(%7d): %s"  % (lap, starttime.strftime("%Y/%m/%d %H:%M:%S")))
        print ("* 終了(%7d): %s"  % (cnt, endtime.strftime("%Y/%m/%d %H:%M:%S")))
        print ("*  black:%7d    vs    white:%7d    (%3.3f)%%  *" % (black_win, white_win, result))
        print ("*********************** %2d / %2d **************************" % (tmi,tmx))
        score_history.append(white_win / pCount * 100)

    #DQN保存
    white_player.save_network("AI_Deep.pkl")

    plt.plot(score_history)
    plt.xlabel("epoch(" + str(epoch)+")数")
    plt.ylabel("勝率(%)")
    #plt.savefig(str(endtime) + ".png")
    #plt.show()


if __name__ == "__main__":
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数

    if(argc > 1 and argvs[1] == "tr"):
        if(argc > 3):
            wCount = int(argvs[2])
            wCount2 = int(argvs[3])
            Training(wCount,False,wCount2)
        elif(argc > 2):
            wCount = int(argvs[2])
            Training(wCount,False)
        else:
            Training()
    else:
        MyAIGame.DisplayBorde()
