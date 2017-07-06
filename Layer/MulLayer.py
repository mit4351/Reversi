##!/usr/nim/python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        result = x*y
        return result

    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

if  __name__ == '__main__':
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数

    app_Layer = MulLayer()
    tax_Layer = MulLayer()
    #forward
    app_price = app_Layer.forward(100,2)   #単価, 個数
    price = tax_Layer.forward(app_price, 1.1)  #金額, 税率
    print("price=%d" %(price))
    #backward
    d_app_price, d_tax = tax_Layer.backward(1) # 値引額
    print("d_app_price=%d" % (d_app_price))
    d_app, d_app_num = app_Layer.backward(d_app_price)
    print("d_app=%3.3f  d_app_num=%3.3f  d_tax=%3.3f" % (d_app, d_app_num, d_tax))
