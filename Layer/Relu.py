##!/usr/nim/python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.layers import Relu

if  __name__ == '__main__':
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数

    relu = Relu()
    x = np.random.uniform(low=-0.1, high=0.1, size=(3, 3))
    out = relu.forward(x)
    dx  = relu.backward(out)

    print("x=%s" %(x))
    print("out=%s" %(out))
    print("dx=%s" %(dx))
