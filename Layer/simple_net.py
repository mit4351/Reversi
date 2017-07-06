# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        print('loss:z=' + str(z))
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

def softmax(x):
    if x.ndim == 2:
       x = x.T
       x = x - np.max(x, axis=0)
       y = np.exp(x) / np.sum(np.exp(x), axis=0)
       return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
       t = t.reshape(1, t.size)
       y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
       t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
   
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
    
        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
    
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()

    return grad


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])
net = simpleNet()
print('-*-'*10 + 'net.W' + '-*-'*10)
print(net.W)

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print('-*-'*10 + 'dW' + '-*-'*10)
print(dW)
