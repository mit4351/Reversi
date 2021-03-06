#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
1.オセロの基本ロジック
2.ランダムで打つ
3.最もたくさん石が取れる手を選ぶ
4.ちょっとだけオセロの定石を知ってる
5.NN思考エンジンを実装  <<--ここ
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

def Training(bord_size,pCount=10,pOutput=True,epoch=2,file_name="NNetWork.pkl"):
    # 盤面を出力するか
    output = pOutput
    #print "開始(%d): %s" % (times, datetime.datetime.today())
    score_history = []

    # プレイヤー
    black_player = Player.RandomAi(ReverseCommon.BLACK, bord_size)
    #black_player = Player.RandomAiKnowGoodMove(ReverseCommon.BLACK, bord_size)
    #black_player = Player.NextStoneMaxAi(ReverseCommon.BLACK, bord_size)
    white_player = Player.NNetWork(ReverseCommon.WHITE, bord_size,file_name, True)
    cnt = 0
    lap = 0
    start = 0
    tmx = 0
    tmi = 0
    otetuki = 0
    for j in range(0, epoch):
        starttime    =  datetime.datetime.today()
        lap = cnt +1
        # 勝利数
        black_win = 0
        white_win = 0
        otetuki = 0
        for i in range(0, pCount):
            # 盤面作成
            reverse_board = ReverseBoard.ReverseBoard(bord_size)

            # ゲーム開始
            game = Game.Game(black_player, white_player, reverse_board)
            player1_score, player2_score = game.play(output)
            cnt = white_player.update_network(player2_score - player1_score)
            if i==0:
                lap = cnt
            if j==0 and i==0:
                start = cnt
            if player1_score > (bord_size * bord_size):
                 otetuki += 1

            # 勝者判定
            if player1_score >  player2_score:
                black_win += 1
            else:
                white_win += 1
            cnt += 1

        # 勝利数
        endtime    =  datetime.datetime.today()
        result = float(white_win) / float(white_win + black_win) * 100
        tmi, tmx = white_player.DTMX()
        print ("****************** 対戦結果 epoch(%5d) **************" % j)
        print ("* 開始(%7d): %s"  % (lap, starttime.strftime("%Y/%m/%d %H:%M:%S")))
        print ("* 終了(%7d): %s"  % (cnt, endtime.strftime("%Y/%m/%d %H:%M:%S")))
        print ("*  black:%7d    vs    white:%7d    (%3.3f)%%  *" % (black_win, white_win, result))
        print ("*********************** %2d / %2d (%5d)****************" % (tmi,tmx,otetuki))
        score_history.append(white_win / pCount * 100)

    #DQN保存
    white_player.save_network(file_name)
    plt.title("%s*%sNN3 (Start:%5d  End:%5d)"  % (bord_size, bord_size, start, cnt))
    plt.plot(score_history)
    plt.xlabel("epoch(" + str(epoch)+")数")
    plt.ylabel("勝率(%)")
    plt.savefig(str(endtime) + ".png")
    plt.show()


if __name__ == "__main__":
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数
    bord_size = 4
    filename = str(bord_size) + "_" +  str(bord_size) + "_" + "NNetWork.pkl"

    if(argc > 1 and argvs[1] == "tr"):
        if(argc > 3):
            wCount = int(argvs[2])
            wCount2 = int(argvs[3])
            Training(bord_size, wCount,False,wCount2,filename)
        elif(argc > 2):
            wCount = int(argvs[2])
            Training(bord_size, wCount,False,filename)
        else:
            Training(bord_size, 1000,True,1,filename)
    else:
        MyAIGame.DisplayBorde(bord_size,filename)
