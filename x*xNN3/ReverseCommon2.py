#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy
import numpy as np
from functions import base10to, base10from

# 定数
NONE = None # 何も置かれていない
WHITE = False # 白
BLACK = True # 黒

def boardToKey3(stone_status):
    bord_size = len(stone_status)
    result = ""
    for i in range(0, bord_size):
        for j in range(0, bord_size):
            if stone_status[i][j] == BLACK:
                result += "1"
            elif stone_status[i][j] == WHITE:
                result += "2"
            else:
                result += "0"
    return base10from(result,3)

def boardToKey2(stone_status):
    bord_size = len(stone_status)
    result = []
    for i in range(0, bord_size):
        for j in range(0, bord_size):
            if stone_status[i][j] == BLACK:
                result.append(0.001)
            elif stone_status[i][j] == WHITE:
                result.append(-0.001)
            else:
                result.append(0.0)
    return result

""" ボードのキー値を返す """
def boardToKey(stone_status):
    bord_size = len(stone_status)
    result = []
    for i in range(0, bord_size):
        for j in range(0, bord_size):
            if stone_status[i][j] == BLACK:
                result.append("B")
            elif stone_status[i][j] == WHITE:
                result.append("W")
            else:
                result.append("N")
    return toString(result)

"""
学習モデルの容量を小さくする為
圧縮を試みる
"""
def toString(pKey):
    result = ""
    cnt = 0
    old = pKey[0]
    for i in range(0,len(pKey)):
        if old == pKey[i]:
            cnt += 1
            if i == len(pKey) -1:
                result += old
                if cnt > 1:
                    result += str(cnt)
        else:
            result += old
            if cnt > 1:
                result += str(cnt)
            old = pKey[i]
            cnt = 1
    return result

# Common Functions
""" 指定した色のインデックス行列を返す """
def get_index(stone_status, color):
    bord_size = len(stone_status)
    result = []
    for i in range(0, bord_size):
        for j in range(0, bord_size):
            if stone_status[i][j] == color:
                result.append(i*bord_size + j +1)
    return result

""" 指定したIndexのPoint座標を返す """
def IndexToPoint(pIndex, bord_size):
    wRow = int((pIndex -1) / int(bord_size))
    wCol = int((pIndex -1) % int(bord_size))
    #print("pIndex=" + str(pIndex) + "::wRow=" + str(wRow) + "::wCol=" + str(wCol) )
    return [wRow, wCol]

def IndexsToPoints(pIndexs, bord_size):
    result = []
    for i in range(0, len(pIndexs)):
        wRow = int((pIndexs[i] -1)/ int(bord_size))
        wCol = int((pIndexs[i] -1)% int(bord_size))
        result.append([wRow, wCol])
    return result

""" 指定したPoint座標のIndexを返す """
def PointToIndex(pPoint, bord_size):
    wRow = pPoint[0] * int(bord_size)
    wCol = pPoint[1] + 1
    wIndex = wRow + wCol
    #print("pPoint=%s:::wIndex=%s"  %(pPoint,wIndex))
    return wIndex

def PointsToIndexs(pPoints, bord_size):
    result = []
    for i in range(0, len(pPoints)):
        wRow = pPoints[i][0] * int(bord_size)
        wCol = pPoints[i][1] + 1
        wIndex = wRow + wCol
        result.append(wIndex)
    return result

def PointsToMask(pPoints, bord_size):
    Indexs = PointsToIndexs(pPoints,bord_size)
    #print("Indexs=%s" % (Indexs))
    #m = np.random.uniform(low=0.0, high=0.2, size=(65))
    p = int(bord_size) * int(bord_size) + 1
    result = np.zeros(p)
    for i in Indexs:
        result[i] = 1 #m[i]

    if len(Indexs) ==0:
        result[0] = 1 #m[0]

    #print("m.shape=%s :: result.shape=%s :: m=%s :: result=%s" % (m.shape,result.shape, m, result))
    return result

""" 指定した色の現在のスコアを返す """
def get_score(stone_status, color):
    bord_size = len(stone_status)
    score = 0
    for i in range(0, bord_size):
        for j in range(0, bord_size):
            if stone_status[i][j] == color:
                score += 1
    return score

""" 何も置かれていない場所の数を返す """
def get_remain(stone_status):
    bord_size = len(stone_status)
    count = 0
    for i in range(0, bord_size):
        for j in range(0, bord_size):
            if stone_status[i][j] is None:
                count += 1
    return count

""" 指定座標の右側に返せる石があるか調べる """
def has_right_reversible_stone(stone_status, i, j, color):
    bord_size = len(stone_status)
    enemy = not(bool(color))
    if j <= int(bord_size / 2) and stone_status[i][j+1] == enemy:
        for k in range(j + 2, bord_size):
            if stone_status[i][k] == color:
                return True
            elif stone_status[i][k] == NONE:
                break
    return False

""" 指定座標の左側に返せる石があるか調べる """
def has_left_reversible_stone(stone_status, i, j, color):
    enemy = not(bool(color))
    if j >=2 and stone_status[i][j-1] == enemy:
        for k in range(j - 2, -1, -1):
            if stone_status[i][k] == color:
                return True
            elif stone_status[i][k] == NONE:
                break
    return False

""" 指定座標の上に返せる石があるか調べる """
def has_upper_reversible_stone(stone_status, i, j, color):
    enemy = not(bool(color))
    if i >= 2 and stone_status[i-1][j] == enemy:
        for k in range(i - 2, -1, -1):
            if stone_status[k][j] == color:
                return True
            elif stone_status[k][j] == NONE:
                break
    return False

""" 指定座標の下に返せる石があるか調べる """
def has_lower_reversible_stone(stone_status, i, j, color):
    bord_size = len(stone_status)
    enemy = not(bool(color))
    if i <= 0  and stone_status[i+1][j] == enemy:
        for k in range(i + 2, bord_size):
            if stone_status[k][j] == color:
                return True
            elif stone_status[k][j] == NONE:
                break
    return False

""" 指定座標の右上に返せる石があるか調べる """
def has_right_upper_reversible_stone(stone_status, i, j, color):
    bord_size = len(stone_status)
    enemy = not(bool(color))
    if i >= 2 and j <= 2 and stone_status[i-1][j+1] == enemy:
        k = 2
        while i - k >= 0 and j + k < bord_size:
            if stone_status[i-k][j+k] == color:
                return True
            elif stone_status[i-k][j+k] == NONE:
                break
            k += 1
    return False

""" 指定座標の左下に返せる石があるか調べる """
def has_left_lower_reversible_stone(stone_status, i, j, color):
    bord_size = len(stone_status)
    enemy = not(bool(color))
    if j >= 2 and i <= 2  and stone_status[i+1][j-1] == enemy:
        k = 2
        while i + k < bord_size and j - k >= 0:
            if stone_status[i+k][j-k] == color:
                return True
            elif stone_status[i+k][j-k] == NONE:
                break
            k += 1
    return False

""" 指定座標の左上に返せる石があるか調べる """
def has_left_upper_reversible_stone(stone_status, i, j, color):
    enemy = not(bool(color))
    if i >= 2 and j >= 2 and stone_status[i-1][j-1] == enemy:
        k = 2
        while i - k >= 0 and j - k >= 0:
            if stone_status[i-k][j-k] == color:
                return True
            elif stone_status[i-k][j-k] == NONE:
                break
            k += 1
    return False

""" 指定座標の右下に返せる石があるか調べる """
def has_right_lower_reversible_stone(stone_status, i, j, color):
    bord_size = len(stone_status)
    enemy = not(color)
    if i <= 0 and j <= 0  and stone_status[i+1][j+1] == enemy:
        k = 2
        while i + k < bord_size and j + k < bord_size:
            if stone_status[i+k][j+k] == color:
                return True
            elif stone_status[i+k][j+k] == NONE:
                break
            k += 1
    return False

""" ゲーム終了か判定する """
def is_game_set(stone_status):
    if len(get_puttable_points(stone_status, WHITE)) == 0 and len(get_puttable_points(stone_status, BLACK)) == 0:
        return True
    return False

""" 指定した場所に置けるかを返す """
def isPuttable_points(pBoard, index):
    result = False
    points = get_puttable_points(pBoard.stone_status, pBoard.CurrentColor)
    result = index in points
    #print("index=" + str(index) + str(result))
    #print(points)
    return result

""" カレント手番が置けるかを返す """
def isPuttable(pBoard):
    points = get_puttable_points(pBoard.stone_status, pBoard.CurrentColor)
    print("isPuttable=" + str(len(points)) +"::"+ str(pBoard.CurrentColor))
    if(len(points)>0):
        return True
    return False


""" 指定した色が置ける座標をすべて返す """
def get_puttable_points(stone_status, color):
    bord_size = len(stone_status)
    #print("bord_size=%s" %(bord_size))
    points = []
    for i in range(0, bord_size):
        for j in range(0, bord_size):
            if stone_status[i][j] != NONE:
                # 何か置かれている場所はする
                continue

            # 左右に走査
            if has_right_reversible_stone(stone_status, i, j, color) or has_left_reversible_stone(stone_status, i, j, color):
                points.append([i, j])
                continue

            # 上下に走査
            if has_upper_reversible_stone(stone_status, i, j, color) or has_lower_reversible_stone(stone_status, i, j, color):
                points.append([i, j])
                continue

            # 右斜め上、左斜め下
            if has_right_upper_reversible_stone(stone_status, i, j, color) or has_left_lower_reversible_stone(stone_status, i, j, color):
                points.append([i, j])
                continue

            # 左上、右下
            if has_left_upper_reversible_stone(stone_status, i, j, color) or has_right_lower_reversible_stone(stone_status, i, j, color):
                points.append([i, j])
                continue
    return points

"""============================================
 ひっくり返す
============================================"""
def put_stone(stone_status, color, i, j):
    new_board = copy.deepcopy(stone_status)
    # 右側をひっくり返しord[i][k] != color:
    if has_right_reversible_stone(new_board, i, j, color):
        k = j + 1
        while new_board[i][k] != color:
            new_board[i][k] = color
            k += 1

    # 左側をひっくり返していく
    if has_left_reversible_stone(new_board, i, j, color):
        k = j - 1
        while new_board[i][k] != color:
            new_board[i][k] = color
            k -= 1

    # 上側をひっくり返していく
    if has_upper_reversible_stone(new_board, i, j, color):
        k = i - 1
        while new_board[k][j] != color:
            new_board[k][j] = color
            k -= 1

    # 下側をひっくり返していく
    if has_lower_reversible_stone(new_board, i, j, color):
        k = i + 1
        while new_board[k][j] != color:
            new_board[k][j] = color
            k += 1

    # 右下をひっくりかえしていく
    if has_right_lower_reversible_stone(new_board, i, j, color):
        k = 1
        while new_board[i+k][j+k] != color:
            new_board[i+k][j+k] = color
            k += 1

    # 左上をひっくりかえしていく
    if has_left_upper_reversible_stone(new_board, i, j, color):
        k = 1
        while new_board[i-k][j-k] != color:
            new_board[i-k][j-k] = color
            k += 1

    # 右上をひっくりかえしていく
    if has_right_upper_reversible_stone(new_board, i, j, color):
        k = 1
        while new_board[i-k][j+k] != color:
            new_board[i-k][j+k] = color
            k += 1

    # 左下をひっくり返していく
    if has_left_lower_reversible_stone(new_board, i, j, color):
        k = 1
        while new_board[i+k][j-k] != color:
            new_board[i+k][j-k] = color
            k += 1

    new_board[i][j] = color
    return new_board

def print_board(board):
    stone_status = board.stone_status
    bord_size = len(stone_status)
    """盤面表示"""
    label = "  "
    for p in range(1, bord_size + 1):
        label += " " + str(p)
    print (label)
    for i in range(0, bord_size):
        row = str(i+1) + " |"
        for j in range(0, bord_size):
            if stone_status[i][j] == NONE:
                row += " "
            elif stone_status[i][j] == WHITE:
                row += "W"
            else:
                row += "B"
            row += "|"
        print (row)
    print ("")
