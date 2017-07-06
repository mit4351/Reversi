# coding: utf-8
import numpy as np

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
