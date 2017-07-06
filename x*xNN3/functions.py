# coding: utf-8
import numpy as np
import sys
import random

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def dtanh(y):
    return 1 - y**2

def dMean_squared_error(y, t):
    return y - t

'''
 シグモイド関数(活性化関数)
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

'''
 ランプ関数(活性化関数)
 Rectified Linear Unit, Rectifier, 正規化線形関数
'''
def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad

def myMax(py,Flg=False):
    m = py.reshape(len(py))
    #print(":m.shape=%s:: m=%s" %(m.shape,m))
    y=m[m!=0]
    #print(":y.shape=%s:: y=%s" %(y.shape,y))
    i = max(y)
    #print("i=%s" %(i))
    #pp = np.nanargmax(py, axis=1)
    pp = np.where(np.array(m) == i)
    p = pp[0][random.randint(0, len(pp[0]) - 1)]
    #print("p=%s" %(p))
    return p

def myMin(py,Flg=False):
    m = py.reshape(len(py))
    #print(":m.shape=%s:: m=%s" %(m.shape,m))
    y=m[m!=0]
    #print(":y.shape=%s:: y=%s" %(y.shape,y))
    i = min(y)
    pp = np.where(np.array(m) == i)
    p = pp[0][random.randint(0, len(pp[0]) - 1)]
    return p

def myRandom(py,Flg=False):
    y=py.copy()
    wk = y.tolist()
    ps = np.where(y[y !=0])
    p  =  ps[random.randint(0, len(ps) - 1)]
    return p


'''
 恒等関数(出力層活性化関数)
 回帰問題
 '''
def identity_function(x):
    return x

'''
 ソフトマックス関数(出力層活性化関数)
 分類問題
'''
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

"""
2乗和誤差(mean squared error)
損失関数(loss function)
"""
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

"""
交差エントロピー誤差(cross entropy error)
損失関数(loss function)
"""
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vector(正解:1 誤答:0)の場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

def init_KeysCount(masuCount=8, doropoutRait=0.4, tailcut=5):
    result = {}
    for i in range(2,masuCount*masuCount):
        if i < masuCount*masuCount - tailcut:
            result[str(i)] = [int(2**i * doropoutRait),0]
        else:
            result[str(i)] = [0,0]
    #print(result)
    return result

#10進数nを、b進数に変換
def base10to(n, b):
    if (int(n/b)):
        return base10to(int(n/b), b) + str(n%b)
    return str(n%b)

#b進数を10進数に変換
def base10from(num, b):
    n = 0
    numlist = list(num)
    while (numlist):
        n *= b
        n += int(numlist.pop(0))
    return n
